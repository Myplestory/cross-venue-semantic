"""
Polymarket WebSocket feed for real-time CLOB price changes.

Subscribes with specific asset (token) IDs. On each price-change event,
re-fetches the full CLOB book via REST (the WS message itself is lightweight
and doesn't contain depth).
"""

import asyncio
import json
import logging
from typing import Optional, Callable, Dict

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spread_scanner import VenueBook, fetch_polymarket_book
from ..compliance.metrics import SystemMetrics

logger = logging.getLogger(__name__)

POLY_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
REST_REFRESH_INTERVAL = 30.0


class PolymarketWSFeed:
    """
    WebSocket feed for Polymarket CLOB price changes.
    
    Subscribes with specific asset (token) IDs. On each price-change
    event, re-fetches the full CLOB book via REST (the WS message itself
    is lightweight and doesn't contain depth). Includes periodic REST
    refresh as safety net.
    """
    
    def __init__(
        self,
        market_id: str,
        token_map: Dict[str, Optional[str]],  # {"yes_token": "...", "no_token": "..."}
        on_orderbook_update: Callable[[VenueBook], None],
        metrics: Optional[SystemMetrics] = None,
    ):
        """
        Initialize Polymarket WebSocket feed.
        
        Args:
            market_id: Polymarket market/condition ID
            token_map: Dict with "yes_token" and "no_token" keys
            on_orderbook_update: Callback when orderbook is updated
            metrics: Optional SystemMetrics for tracking
        """
        self.market_id = market_id
        self.yes_token = token_map.get("yes_token")
        self.no_token = token_map.get("no_token")
        self.on_orderbook_update = on_orderbook_update
        self.metrics = metrics
        self._running = False
        self._ws = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.update_count = 0
    
    async def run(self, session: aiohttp.ClientSession):
        """
        Connect and subscribe to token IDs.
        
        Runs WebSocket loop and periodic REST refresh in parallel.
        
        Args:
            session: aiohttp ClientSession for REST orderbook fetches
        """
        self._running = True
        self._session = session
        
        # Bootstrap: fetch initial orderbook via REST
        try:
            book = await fetch_polymarket_book(session, self.market_id)
            if book:
                await self._notify_update(book)
        except Exception as e:
            logger.warning(f"[Poly WS] Bootstrap fetch failed: {e}")
        
        # Run WebSocket and REST refresh in parallel
        await asyncio.gather(
            self._ws_loop(),
            self._rest_refresh_loop(),
            return_exceptions=True,
        )
    
    async def stop(self):
        """Stop the WebSocket feed."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
    
    async def _ws_loop(self):
        """WebSocket connection and message handling loop."""
        all_tokens = [t for t in [self.yes_token, self.no_token] if t]
        if not all_tokens:
            logger.warning("[Poly WS] No token IDs — WS feed inactive")
            return
        
        while self._running:
            try:
                logger.info(f"[Poly WS] Connecting to {POLY_WS_URL} ({len(all_tokens)} tokens)")
                async with websockets.connect(
                    POLY_WS_URL, ping_interval=25, ping_timeout=15,
                ) as ws:
                    self._ws = ws
                    logger.info("[Poly WS] Connected — subscribing")
                    
                    # Polymarket CLOB subscription format
                    await ws.send(json.dumps({
                        "type": "market",
                        "assets_ids": all_tokens,
                    }))
                    
                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle_message(raw)
            
            except ConnectionClosed as e:
                logger.warning(f"[Poly WS] Connection closed: {e}. Reconnecting in 5s...")
                if self.metrics:
                    await self.metrics.increment_websocket_reconnect("polymarket")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[Poly WS] Error: {e}. Reconnecting in 10s...")
                if self.metrics:
                    await self.metrics.increment_api_error("polymarket_ws")
                await asyncio.sleep(10)
    
    async def _handle_message(self, raw: str):
        """Handle incoming WebSocket message."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        
        # Polymarket CLOB WS sends either a single dict or a list of dicts
        items: list[dict] = []
        if isinstance(payload, list):
            items = [d for d in payload if isinstance(d, dict)]
        elif isinstance(payload, dict):
            items = [payload]
        else:
            return
        
        # Check if any of our tokens changed
        token_changed = False
        for data in items:
            asset_id = (
                data.get("asset_id")
                or data.get("token_id")
                or data.get("market")
                or ""
            )
            if asset_id in [self.yes_token, self.no_token]:
                token_changed = True
                break
        
        # Refetch full orderbook if our tokens changed
        if token_changed and self._session:
            try:
                book = await fetch_polymarket_book(self._session, self.market_id)
                if book:
                    await self._notify_update(book)
                    self.update_count += 1
            except Exception as exc:
                logger.debug(f"[Poly WS] Refetch failed for {self.market_id[:20]}: {exc}")
                if self.metrics:
                    await self.metrics.increment_api_error("polymarket_ws")
    
    async def _notify_update(self, book: VenueBook):
        """Notify orderbook manager of update."""
        try:
            if asyncio.iscoroutinefunction(self.on_orderbook_update):
                await self.on_orderbook_update(book)
            else:
                self.on_orderbook_update(book)
        except Exception as e:
            logger.error(f"[Poly WS] Error in orderbook update callback: {e}", exc_info=True)
    
    async def _rest_refresh_loop(self):
        """Periodic REST refresh as safety net for consistency."""
        while self._running:
            await asyncio.sleep(REST_REFRESH_INTERVAL)
            if not self._running or not self._session:
                break
            
            try:
                logger.debug(f"[Poly REST] Periodic refresh for {self.market_id[:20]}")
                book = await fetch_polymarket_book(self._session, self.market_id)
                if book:
                    await self._notify_update(book)
            except Exception as exc:
                logger.debug(f"[Poly REST] Refresh failed: {exc}")
                if self.metrics:
                    await self.metrics.increment_api_error("polymarket_rest")

