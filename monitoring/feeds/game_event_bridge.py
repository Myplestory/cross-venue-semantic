"""
High-frequency game event polling bridge for latency measurement.

Polls REST APIs (Riot/BALLDONTLIE) at high frequency and pushes events via
WebSocket or direct callback to minimize latency between game events and
prediction market updates.

Optimized for:
- Sub-second polling intervals (500ms default, configurable)
- Precise timestamp tracking (UTC with microsecond precision)
- Integration with LatencyCorrelationEngine
- Real-time push to connected clients
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Callable, Dict, List, Optional, Set, Any

import aiohttp
import websockets
from websockets.server import serve

from ..compliance.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from ..compliance.metrics import SystemMetrics
from ..compliance.audit_logger import AuditLogger
from .riot_api import RiotAPIClient, RiotEventPoller

logger = logging.getLogger(__name__)


@dataclass
class GameEvent:
    """Structured game event with precise timestamps."""
    event_type: str
    event_timestamp: datetime  # When event actually occurred (from API)
    ingestion_timestamp: datetime  # When we received it (for latency measurement)
    source: str  # 'riot_api', 'balldontlie', 'polymarket_sports'
    match_id: str
    event_data: Dict[str, Any]
    raw_payload: Optional[Dict[str, Any]] = None


class GameEventBridge:
    """
    High-frequency polling bridge for game events.
    
    Polls REST APIs at configurable intervals (default 500ms) and pushes
    events to connected clients via WebSocket or direct callbacks.
    
    Supports:
    - Riot API (match timeline polling)
    - BALLDONTLIE API (future support)
    - Direct callback integration with LatencyCorrelationEngine
    """
    
    def __init__(
        self,
        riot_api_key: Optional[str] = None,
        balldontlie_api_key: Optional[str] = None,
        match_ids: Optional[Set[str]] = None,
        poll_interval_ms: float = 500.0,  # 500ms = 2 polls/second
        ws_host: str = "localhost",
        ws_port: int = 8765,
        on_event: Optional[Callable[[GameEvent], None]] = None,
        metrics: Optional[SystemMetrics] = None,
        audit_logger: Optional[AuditLogger] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """
        Initialize game event bridge.
        
        Args:
            riot_api_key: Riot Games API key
            balldontlie_api_key: BALLDONTLIE API key (future support)
            match_ids: Set of match IDs to poll
            poll_interval_ms: Polling interval in milliseconds (default 500ms)
            ws_host: WebSocket server host
            ws_port: WebSocket server port
            on_event: Direct callback for events (bypasses WebSocket)
            metrics: SystemMetrics instance
            audit_logger: AuditLogger instance
            circuit_breaker: CircuitBreaker instance
        """
        self.riot_api_key = riot_api_key
        self.balldontlie_api_key = balldontlie_api_key
        self.match_ids = match_ids or set()
        self.poll_interval_ms = poll_interval_ms
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.on_event = on_event
        self.metrics = metrics
        self.audit_logger = audit_logger
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        
        # State
        self._running = False
        self._server: Optional[Any] = None
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.processed_event_ids: Set[str] = field(default_factory=set)
        self.event_history: deque = deque(maxlen=1000)  # Keep last 1000 events
        
        # Riot API client
        self.riot_client: Optional[RiotAPIClient] = None
        self.riot_session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.total_events_polled = 0
        self.total_events_pushed = 0
        self.last_poll_timestamp: Optional[datetime] = None
        self.avg_poll_latency_ms = 0.0
    
    async def start(self):
        """Start WebSocket server and background polling."""
        self._running = True
        
        # Initialize Riot API client if key provided
        if self.riot_api_key:
            self.riot_session = aiohttp.ClientSession()
            self.riot_client = RiotAPIClient(
                api_key=self.riot_api_key,
                circuit_breaker=self.circuit_breaker,
                metrics=self.metrics,
            )
            # Initialize session
            await self.riot_client.__aenter__()
            logger.info("[GameEventBridge] Riot API client initialized")
        
        # Start WebSocket server (if not using direct callback)
        if not self.on_event:
            self._server = await serve(
                self._handle_client,
                self.ws_host,
                self.ws_port
            )
            logger.info(
                f"[GameEventBridge] WebSocket server started on "
                f"ws://{self.ws_host}:{self.ws_port}"
            )
        
        # Start background polling
        asyncio.create_task(self._poll_loop())
        logger.info(
            f"[GameEventBridge] Started polling loop "
            f"(interval: {self.poll_interval_ms}ms)"
        )
    
    async def stop(self):
        """Stop server and close all connections."""
        self._running = False
        
        # Close WebSocket server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        # Close all client connections
        for client in list(self.connected_clients):
            try:
                await client.close()
            except Exception:
                pass
        self.connected_clients.clear()
        
        # Close Riot API client
        if self.riot_client:
            await self.riot_client.__aexit__(None, None, None)
        if self.riot_session:
            await self.riot_session.close()
        
        logger.info("[GameEventBridge] Stopped")
    
    async def _handle_client(self, ws: websockets.WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connection."""
        self.connected_clients.add(ws)
        logger.info(
            f"[GameEventBridge] Client connected. "
            f"Total clients: {len(self.connected_clients)}"
        )
        
        try:
            # Send welcome message with current state
            await ws.send(json.dumps({
                "type": "connected",
                "message": "Connected to Game Event Bridge",
                "match_ids": list(self.match_ids),
                "poll_interval_ms": self.poll_interval_ms,
                "stats": {
                    "total_events_polled": self.total_events_polled,
                    "total_events_pushed": self.total_events_pushed,
                    "avg_poll_latency_ms": self.avg_poll_latency_ms,
                },
            }))
            
            # Keep connection alive and handle client messages
            async for message in ws:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
                    elif data.get("type") == "subscribe":
                        # Client can subscribe to specific match IDs
                        match_ids = data.get("match_ids", [])
                        if match_ids:
                            self.match_ids.update(match_ids)
                            await ws.send(json.dumps({
                                "type": "subscribed",
                                "match_ids": list(self.match_ids),
                            }))
                except json.JSONDecodeError:
                    pass
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(ws)
            logger.info(
                f"[GameEventBridge] Client disconnected. "
                f"Total clients: {len(self.connected_clients)}"
            )
    
    async def _poll_loop(self):
        """Background loop that polls APIs and pushes events."""
        while self._running:
            poll_start = time.perf_counter()
            
            try:
                # Poll Riot API if configured
                if self.riot_client and self.match_ids:
                    await self._poll_riot_api()
                
                # Poll BALLDONTLIE API if configured (future)
                # if self.balldontlie_api_key and self.match_ids:
                #     await self._poll_balldontlie_api()
                
                # Calculate poll latency
                poll_end = time.perf_counter()
                poll_latency_ms = (poll_end - poll_start) * 1000
                
                # Update average poll latency (exponential moving average)
                if self.avg_poll_latency_ms == 0:
                    self.avg_poll_latency_ms = poll_latency_ms
                else:
                    self.avg_poll_latency_ms = (
                        self.avg_poll_latency_ms * 0.9 + poll_latency_ms * 0.1
                    )
                
                self.last_poll_timestamp = datetime.now(UTC)
                
                # Sleep for remaining interval
                sleep_time = max(0, (self.poll_interval_ms / 1000) - (poll_end - poll_start))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[GameEventBridge] Poll error: {e}",
                    exc_info=True
                )
                if self.metrics:
                    await self.metrics.increment_api_error("game_event_bridge_poll")
                await asyncio.sleep(self.poll_interval_ms / 1000)
    
    async def _poll_riot_api(self):
        """
        Poll Riot API for match timeline events.
        
        Handles rate limits (429) with automatic backoff using Retry-After header.
        """
        if not self.riot_client or not self.match_ids:
            return
        
        for match_id in self.match_ids:
            try:
                # Poll match timeline
                timeline = await self.riot_client.get_match_timeline(match_id)
                if not timeline:
                    continue
                
                # Extract events from timeline
                events = self._extract_riot_events(timeline, match_id)
                
                for event in events:
                    await self._process_event(event)
            
            except CircuitBreakerOpenError:
                logger.warning(
                    f"[GameEventBridge] Circuit breaker open for match {match_id}"
                )
                continue
            except aiohttp.ClientResponseError as e:
                # Handle rate limit (429) with Retry-After
                if e.status == 429:
                    retry_after = int(e.headers.get("Retry-After", "1"))
                    logger.warning(
                        f"[GameEventBridge] Rate limited for {match_id}. "
                        f"Waiting {retry_after}s before retry"
                    )
                    if self.metrics:
                        await self.metrics.increment_api_error("riot_api_rate_limit")
                    # Wait before continuing to next match
                    await asyncio.sleep(retry_after)
                else:
                    logger.debug(
                        f"[GameEventBridge] HTTP error {e.status} for {match_id}: {e}"
                    )
                    if self.metrics:
                        await self.metrics.increment_api_error("riot_api_poll")
            except Exception as e:
                logger.debug(
                    f"[GameEventBridge] Error polling Riot API for {match_id}: {e}"
                )
                if self.metrics:
                    await self.metrics.increment_api_error("riot_api_poll")
    
    def _extract_riot_events(
        self,
        timeline: Dict[str, Any],
        match_id: str
    ) -> List[GameEvent]:
        """Extract significant events from Riot timeline."""
        events = []
        frames = timeline.get("info", {}).get("frames", [])
        
        for frame in frames:
            frame_events = frame.get("events", [])
            for event_data in frame_events:
                event_type = event_data.get("type", "")
                
                # Filter significant events
                if event_type not in [
                    "CHAMPION_KILL",
                    "BUILDING_KILL",
                    "ELITE_MONSTER_KILL",
                    "DRAGON_KILL",
                    "BARON_KILL",
                    "INHIBITOR_KILL",
                ]:
                    continue
                
                # Extract timestamp (Riot provides timestamp in milliseconds)
                timestamp_ms = event_data.get("timestamp", 0)
                event_timestamp = datetime.fromtimestamp(
                    timestamp_ms / 1000, UTC
                )
                
                # Create unique event ID
                event_id = (
                    f"{match_id}-{timestamp_ms}-{event_type}-"
                    f"{event_data.get('killerId', '')}-"
                    f"{event_data.get('victimId', '')}"
                )
                
                # Skip if already processed
                if event_id in self.processed_event_ids:
                    continue
                
                self.processed_event_ids.add(event_id)
                
                # Create GameEvent with precise timestamps
                ingestion_timestamp = datetime.now(UTC)
                event = GameEvent(
                    event_type=event_type,
                    event_timestamp=event_timestamp,  # When event occurred
                    ingestion_timestamp=ingestion_timestamp,  # When we received it
                    source="riot_api",
                    match_id=match_id,
                    event_data=event_data,
                    raw_payload=timeline,
                )
                
                events.append(event)
                self.total_events_polled += 1
        
        return events
    
    async def _process_event(self, event: GameEvent):
        """Process a game event and push to clients/callbacks."""
        # Add to history
        self.event_history.append(event)
        
        # Calculate ingestion latency (time from event to ingestion)
        ingestion_latency_ms = (
            (event.ingestion_timestamp - event.event_timestamp).total_seconds() * 1000
        )
        
        # Prepare event payload
        payload = {
            "type": "game_event",
            "event_type": event.event_type,
            "event_timestamp": event.event_timestamp.isoformat(),
            "ingestion_timestamp": event.ingestion_timestamp.isoformat(),
            "ingestion_latency_ms": ingestion_latency_ms,
            "source": event.source,
            "match_id": event.match_id,
            "event_data": event.event_data,
        }
        
        # Push via direct callback (preferred for latency measurement)
        if self.on_event:
            try:
                if asyncio.iscoroutinefunction(self.on_event):
                    await self.on_event(event)
                else:
                    self.on_event(event)
            except Exception as e:
                logger.error(
                    f"[GameEventBridge] Error in event callback: {e}",
                    exc_info=True
                )
        
        # Also broadcast via WebSocket (if clients connected)
        if self.connected_clients:
            await self._broadcast(payload)
        
        # Log for audit
        if self.audit_logger:
            self.audit_logger.log_system_event(
                "game_event_received",
                {
                    "event_type": event.event_type,
                    "match_id": event.match_id,
                    "ingestion_latency_ms": ingestion_latency_ms,
                }
            )
        
        # Update metrics
        if self.metrics:
            await self.metrics.increment_game_events()
        
        self.total_events_pushed += 1
        
        logger.debug(
            f"[GameEventBridge] Event: {event.event_type} | "
            f"Match: {event.match_id} | "
            f"Ingestion latency: {ingestion_latency_ms:.1f}ms"
        )
    
    async def _broadcast(self, payload: Dict[str, Any]):
        """Broadcast event to all connected WebSocket clients."""
        if not self.connected_clients:
            return
        
        json_msg = json.dumps(payload)
        disconnected = []
        
        for client in self.connected_clients:
            try:
                await client.send(json_msg)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
            except Exception as e:
                logger.debug(
                    f"[GameEventBridge] Error sending to client: {e}"
                )
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.connected_clients.discard(client)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "running": self._running,
            "connected_clients": len(self.connected_clients),
            "match_ids": list(self.match_ids),
            "poll_interval_ms": self.poll_interval_ms,
            "total_events_polled": self.total_events_polled,
            "total_events_pushed": self.total_events_pushed,
            "avg_poll_latency_ms": self.avg_poll_latency_ms,
            "last_poll_timestamp": (
                self.last_poll_timestamp.isoformat()
                if self.last_poll_timestamp else None
            ),
            "event_history_size": len(self.event_history),
        }

