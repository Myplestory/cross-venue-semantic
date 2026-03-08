"""
Kalshi WebSocket feed for real-time orderbook updates.

Subscribes to orderbook_delta for specific tickers and applies incremental
updates to orderbooks. Integrates with circuit breaker for resilience.
"""

import asyncio
import json
import logging
from typing import Optional, Callable

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from discovery.kalshi_poller import _load_kalshi_private_key, _sign_kalshi_request
from spread_scanner import VenueBook, BookLevel, fetch_kalshi_book
from ..compliance.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from ..compliance.metrics import SystemMetrics

import config

logger = logging.getLogger(__name__)

# Kalshi WebSocket URL
KALSHI_WS_URL = (
    "wss://demo-api.kalshi.co/trade-api/ws/v2"
    if config.KALSHI_USE_DEMO
    else "wss://api.elections.kalshi.com/trade-api/ws/v2"
)
KALSHI_WS_SIGN_PATH = "/trade-api/ws/v2"

# REST refresh interval (seconds) - safety net for consistency
REST_REFRESH_INTERVAL = 30.0


class KalshiWSFeed:
    """
    WebSocket feed for Kalshi orderbook deltas.
    
    Subscribes to orderbook_delta for a specific ticker.
    On each delta, applies changes to the in-memory book and notifies
    the orderbook manager. Includes periodic REST refresh as safety net.
    """
    
    def __init__(
        self,
        ticker: str,
        on_orderbook_update: Callable[[VenueBook], None],
        circuit_breaker: Optional[CircuitBreaker] = None,
        metrics: Optional[SystemMetrics] = None,
        api_key_id: Optional[str] = None,
        private_key=None,
    ):
        """
        Initialize Kalshi WebSocket feed.
        
        Args:
            ticker: Kalshi market ticker (e.g., "KXLOLGAME-26FEB22DKT1")
            on_orderbook_update: Callback when orderbook is updated
            circuit_breaker: Optional circuit breaker for resilience
            metrics: Optional SystemMetrics for tracking
            api_key_id: Kalshi API key ID
            private_key: Kalshi private key (loaded from config if None)
        """
        self.ticker = ticker
        self.on_orderbook_update = on_orderbook_update
        self.circuit_breaker = circuit_breaker
        self.metrics = metrics
        self.api_key_id = api_key_id or config.KALSHI_API_KEY_ID
        self.private_key = private_key
        if not self.private_key:
            self.private_key = _load_kalshi_private_key(
                path=config.KALSHI_PRIVATE_KEY_PATH,
                pem=config.KALSHI_PRIVATE_KEY
            )
        self._running = False
        self._ws = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.update_count = 0
        self.current_book: Optional[VenueBook] = None
    
    def _auth_headers(self) -> dict:
        """RSA-PSS signed headers for the WS handshake."""
        if not self.api_key_id or not self.private_key:
            return {}
        result = _sign_kalshi_request(self.private_key, "GET", KALSHI_WS_SIGN_PATH)
        if not result:
            return {}
        sig, ts = result
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }
    
    async def run(self, session: aiohttp.ClientSession):
        """
        Connect and subscribe to orderbook deltas.
        
        Runs WebSocket loop and periodic REST refresh in parallel.
        
        Args:
            session: aiohttp ClientSession for REST fallback
        """
        self._running = True
        self._session = session
        
        # Bootstrap: fetch initial orderbook via REST
        try:
            book = await fetch_kalshi_book(
                session, self.ticker, self.api_key_id, self.private_key
            )
            if book:
                self.current_book = book
                await self._notify_update(book)
        except Exception as e:
            logger.warning(f"[Kalshi WS] Bootstrap fetch failed: {e}")
        
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
        while self._running:
            try:
                headers = self._auth_headers()
                kw = {"ping_interval": 20, "ping_timeout": 10}
                if headers:
                    kw["additional_headers"] = headers
                
                # Connect to WebSocket
                async with websockets.connect(KALSHI_WS_URL, **kw) as ws:
                    await self._ws_handler(ws)
            
            except ConnectionClosed as e:
                logger.warning(f"[Kalshi WS] Connection closed: {e}. Reconnecting in 5s...")
                if self.metrics:
                    await self.metrics.increment_websocket_reconnect("kalshi")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[Kalshi WS] Error: {e}. Reconnecting in 10s...")
                if self.metrics:
                    await self.metrics.increment_api_error("kalshi_ws")
                await asyncio.sleep(10)
    
    async def _ws_handler(self, ws):
        """Handle WebSocket connection and messages."""
        self._ws = ws
        logger.info(f"[Kalshi WS] Connected — subscribing to {self.ticker}")
        
        # Subscribe to orderbook_delta
        await ws.send(json.dumps({
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": [self.ticker],
            },
        }))
        
        async for raw in ws:
            if not self._running:
                break
            await self._handle_message(raw)
    
    async def _handle_message(self, raw: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        
        msg_type = data.get("type", "")
        
        if msg_type == "subscribed":
            logger.info(f"[Kalshi WS] Subscription confirmed: {data.get('msg', {}).get('channel', '')}")
            return
        
        if msg_type == "error":
            logger.warning(f"[Kalshi WS] Server error: {data.get('msg')}")
            if self.metrics:
                await self.metrics.increment_api_error("kalshi_ws")
            return
        
        if msg_type != "orderbook_delta":
            return
        
        msg = data.get("msg", {})
        ticker = msg.get("market_ticker", "")
        if ticker != self.ticker:
            return
        
        yes_delta = msg.get("yes", [])
        no_delta = msg.get("no", [])
        
        # Apply delta to existing book or fetch full book
        if self.current_book:
            updated = self._apply_delta(self.current_book, yes_delta, no_delta)
            self.current_book = updated
            await self._notify_update(updated)
        else:
            # Bootstrap: fetch full book via REST
            if self._session:
                book = await fetch_kalshi_book(
                    self._session, self.ticker, self.api_key_id, self.private_key
                )
                if book:
                    self.current_book = book
                    await self._notify_update(book)
        
        self.update_count += 1
    
    def _apply_delta(
        self, existing: VenueBook, yes_delta: list, no_delta: list
    ) -> VenueBook:
        """
        Apply Kalshi orderbook delta to an existing book.
        
        Kalshi orderbook_delta sends raw bid-side updates for YES and NO.
        The delta format is [[price_cents, new_qty], ...]. A qty of 0 means
        "remove that price level".
        
        Since our arb logic only needs asks, and Kalshi's orderbook semantics
        are YES asks = inverted NO bids and vice-versa, we apply deltas to
        the opposite side's asks.
        """
        def _apply(asks: list[BookLevel], delta: list, invert: bool) -> list[BookLevel]:
            level_map = {round(a.price, 6): a.size for a in asks}
            for entry in delta:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                raw_price = float(entry[0]) / 100.0
                size = float(entry[1])
                price = round(1.0 - raw_price, 6) if invert else raw_price
                if size <= 0:
                    level_map.pop(price, None)
                elif 0 < price < 1:
                    level_map[price] = size
            result = [BookLevel(price=p, size=s) for p, s in level_map.items() if s > 0]
            result.sort(key=lambda l: l.price)
            return result
        
        # YES asks are derived from inverted NO bids → apply no_delta
        # NO asks are derived from inverted YES bids → apply yes_delta
        return VenueBook(
            venue=existing.venue,
            venue_market_id=existing.venue_market_id,
            yes_asks=_apply(existing.yes_asks, no_delta, invert=True),
            yes_bids=existing.yes_bids,
            no_asks=_apply(existing.no_asks, yes_delta, invert=True),
            no_bids=existing.no_bids,
            source="ws_delta",
            resolution_date=existing.resolution_date,
        )
    
    async def _notify_update(self, book: VenueBook):
        """Notify orderbook manager of update."""
        try:
            if asyncio.iscoroutinefunction(self.on_orderbook_update):
                await self.on_orderbook_update(book)
            else:
                self.on_orderbook_update(book)
        except Exception as e:
            logger.error(f"[Kalshi WS] Error in orderbook update callback: {e}", exc_info=True)
    
    async def _rest_refresh_loop(self):
        """Periodic REST refresh as safety net for consistency."""
        while self._running:
            await asyncio.sleep(REST_REFRESH_INTERVAL)
            if not self._running or not self._session:
                break
            
            try:
                logger.debug(f"[Kalshi REST] Periodic refresh for {self.ticker}")
                book = await fetch_kalshi_book(
                    self._session, self.ticker, self.api_key_id, self.private_key
                )
                if book:
                    self.current_book = book
                    await self._notify_update(book)
            except Exception as exc:
                logger.debug(f"[Kalshi REST] Refresh failed: {exc}")
                if self.metrics:
                    await self.metrics.increment_api_error("kalshi_rest")

