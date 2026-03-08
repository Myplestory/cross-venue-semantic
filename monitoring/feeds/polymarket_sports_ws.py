"""
Polymarket Sports WebSocket feed for live game state updates.

Connects to Polymarket's Sports WebSocket API to receive real-time game state
updates including score changes, period transitions, and match status.
No authentication required.
"""

import asyncio
import json
import logging
from typing import Optional, Callable
from datetime import datetime, UTC

import websockets
from websockets.exceptions import ConnectionClosed

from ..compliance.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from ..compliance.metrics import SystemMetrics

logger = logging.getLogger(__name__)

POLYMARKET_SPORTS_WS_URL = "wss://sports-api.polymarket.com/ws"


class PolymarketSportsFeed:
    """
    Polymarket Sports WebSocket feed for live game state.
    
    No authentication required. Receives all active sport events.
    Server sends ping every 5 seconds; respond with pong within 10 seconds
    or connection closes.
    
    Message format:
    {
        "gameId": 1317359,
        "leagueAbbreviation": "lol",
        "slug": "lol-t1-dk-2026-02-22",
        "status": "InProgress",
        "score": "1-0|2-0|Bo5",
        "period": "2/5",
        "live": true
    }
    """
    
    def __init__(
        self,
        market_slug: str,  # e.g., "lol-t1-dk-2026-02-22"
        on_game_state_change: Callable[[dict], None],
        circuit_breaker: Optional[CircuitBreaker] = None,
        metrics: Optional[SystemMetrics] = None,
    ):
        """
        Initialize Polymarket Sports WebSocket feed.
        
        Args:
            market_slug: Market slug to filter events (e.g., "lol-t1-dk-2026-02-22")
            on_game_state_change: Callback when game state changes
            circuit_breaker: Optional circuit breaker for resilience
            metrics: Optional SystemMetrics for tracking
        """
        self.market_slug = market_slug
        self.on_game_state_change = on_game_state_change
        self.circuit_breaker = circuit_breaker
        self.metrics = metrics
        self._running = False
        self._ws = None
        self.last_ping_time: Optional[float] = None
        self.update_count = 0
    
    async def run(self):
        """
        Connect and listen for game state updates.
        
        Handles ping/pong heartbeat automatically.
        Filters messages for the specified market_slug.
        """
        self._running = True
        
        while self._running:
            try:
                logger.info(f"[Poly Sports WS] Connecting to {POLYMARKET_SPORTS_WS_URL}")
                
                # Connect to WebSocket
                async with websockets.connect(
                    POLYMARKET_SPORTS_WS_URL,
                    ping_interval=None,  # We handle pings manually
                    ping_timeout=10,
                ) as ws:
                    await self._ws_handler(ws)
            
            except ConnectionClosed as e:
                logger.warning(f"[Poly Sports WS] Connection closed: {e}. Reconnecting in 5s...")
                if self.metrics:
                    await self.metrics.increment_websocket_reconnect("polymarket_sports")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[Poly Sports WS] Error: {e}. Reconnecting in 10s...")
                if self.metrics:
                    await self.metrics.increment_api_error("polymarket_sports_ws")
                await asyncio.sleep(10)
    
    async def _ws_handler(self, ws):
        """Handle WebSocket connection and messages."""
        self._ws = ws
        logger.info("[Poly Sports WS] Connected — listening for game state updates")
        
        async for raw in ws:
            if not self._running:
                break
            
            try:
                msg = json.loads(raw)
                
                # Handle ping/pong
                if msg.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                    self.last_ping_time = asyncio.get_event_loop().time()
                    continue
                
                # Filter for our market
                slug = msg.get("slug")
                if slug == self.market_slug:
                    await self._handle_game_state(msg)
            
            except json.JSONDecodeError:
                logger.warning("[Poly Sports WS] Invalid JSON received")
            except Exception as e:
                logger.error(f"[Poly Sports WS] Error handling message: {e}", exc_info=True)
    
    async def _handle_game_state(self, msg: dict):
        """Process game state update."""
        event = {
            "source": "polymarket_sports",
            "timestamp": datetime.now(UTC),
            "game_id": msg.get("gameId"),
            "league": msg.get("leagueAbbreviation"),
            "slug": msg.get("slug"),
            "status": msg.get("status"),  # "Scheduled", "InProgress", "Final", etc.
            "score": msg.get("score"),
            "period": msg.get("period"),  # e.g., "2/5" for Bo5
            "live": msg.get("live", False),
            "raw": msg,
        }
        
        self.update_count += 1
        
        # Notify callback
        try:
            if asyncio.iscoroutinefunction(self.on_game_state_change):
                await self.on_game_state_change(event)
            else:
                self.on_game_state_change(event)
        except Exception as e:
            logger.error(
                f"[Poly Sports WS] Error in game state change callback: {e}",
                exc_info=True
            )
        
        logger.debug(
            f"[Poly Sports WS] Game state update: {event['status']} | "
            f"score={event['score']} | period={event['period']}"
        )
    
    async def stop(self):
        """Stop the WebSocket feed."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

