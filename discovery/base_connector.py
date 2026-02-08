"""
Base connector implementation with common WebSocket handling.

Provides reconnection logic, error handling, and event streaming.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, AsyncIterator, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from websockets.client import WebSocketClientProtocol
from abc import ABC, abstractmethod

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .types import VenueType, MarketEvent


logger = logging.getLogger(__name__)


class BaseVenueConnector(ABC):
    """Base class for venue connectors with common WebSocket handling."""
    
    def __init__(
        self,
        venue_name: VenueType,
        ws_url: str,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: Optional[int] = None,
    ):
        """
        Initialize base connector.
        
        Args:
            venue_name: Venue identifier
            ws_url: WebSocket URL to connect to
            reconnect_delay: Seconds to wait before reconnecting
            max_reconnect_attempts: Max reconnect attempts (None = infinite)
        """
        self.venue_name = venue_name
        self.ws_url = ws_url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._ws: Optional[Any] = None  # WebSocketClientProtocol
        self._running = False
        self._reconnect_attempts = 0
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)  # Buffer for non-blocking consumption
        self._receive_task: Optional[asyncio.Task] = None  # Background message reception task
    
    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._ws is not None and self._ws.close_code is None:
            logger.info(f"[{self.venue_name}] Already connected")
            return
        
        try:
            if websockets is None:
                raise ImportError("websockets library not installed")
            
            logger.info(f"[{self.venue_name}] Connecting to {self.ws_url}")
            self._ws = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
            )
            self._reconnect_attempts = 0
            logger.info(f"[{self.venue_name}] Connected successfully")
            
            # Send initial subscription message
            await self._send_subscription()
            
        except Exception as e:
            logger.error(f"[{self.venue_name}] Connection failed: {e}")
            raise
    
    async def start(self) -> None:
        """
        Start background message reception.
        
        This must be called before stream_events() to enable non-blocking consumption.
        The background task will receive messages from WebSocket and enqueue them.
        """
        if self._receive_task is not None and not self._receive_task.done():
            logger.debug(f"[{self.venue_name}] Background task already running")
            return
        
        # Ensure connection
        if self._ws is None or getattr(self._ws, 'close_code', None) is not None:
            await self.connect()
        
        # Start background message reception
        self._running = True
        self._receive_task = asyncio.create_task(self._receive_messages())
        logger.info(f"[{self.venue_name}] Started background message reception")
    
    async def disconnect(self) -> None:
        """Close WebSocket connection and stop background tasks."""
        self._running = False
        
        # Cancel background task
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[{self.venue_name}] Error cancelling receive task: {e}")
            finally:
                self._receive_task = None
        
        # Close WebSocket
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"[{self.venue_name}] Error closing WebSocket: {e}")
            self._ws = None
        
        logger.info(f"[{self.venue_name}] Disconnected")
    
    async def _receive_messages(self) -> None:
        """
        Background task: receive messages from WebSocket and enqueue them.
        
        This runs independently in a background task, allowing the consumer
        to check flags and handle timeouts without blocking.
        """
        while self._running:
            try:
                # Ensure connection
                if self._ws is None or getattr(self._ws, 'close_code', None) is not None:
                    await self._reconnect()
                
                # Receive messages and enqueue (blocks here, but in background)
                async for message in self._ws:
                    if not self._running:
                        break
                    
                    # Put message in queue (non-blocking if queue has space)
                    try:
                        await self._event_queue.put(message)
                    except Exception as e:
                        logger.error(f"[{self.venue_name}] Error enqueueing message: {e}")
                
            except ConnectionClosed:
                logger.warning(f"[{self.venue_name}] Connection closed, reconnecting...")
                await self._reconnect()
            except WebSocketException as e:
                logger.error(f"[{self.venue_name}] WebSocket error: {e}")
                await self._reconnect()
            except asyncio.CancelledError:
                logger.debug(f"[{self.venue_name}] Receive task cancelled")
                break
            except Exception as e:
                logger.error(f"[{self.venue_name}] Unexpected error in receive task: {e}")
                await asyncio.sleep(self.reconnect_delay)
    
    async def stream_events(self) -> AsyncIterator[MarketEvent]:
        """
        Stream market events as they arrive (non-blocking).
        
        Consumes events from the internal queue, allowing periodic flag checks
        and proper cancellation. Background task handles WebSocket reception.
        
        Note: Call start() before using this method for non-blocking behavior.
        """
        # Ensure background task is running
        if self._receive_task is None or self._receive_task.done():
            await self.start()
        
        # Consume from queue with timeout (allows checking _running flag)
        while self._running or not self._event_queue.empty():
            try:
                # Get message from queue with timeout (non-blocking check)
                message = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1  # Check flag every 100ms
                )
                
                # Parse and yield event
                try:
                    event = await self._parse_message(message)
                    if event:
                        yield event
                except Exception as e:
                    logger.error(f"[{self.venue_name}] Error parsing message: {e}")
                    logger.debug(f"[{self.venue_name}] Message: {message[:200]}")
                
            except asyncio.TimeoutError:
                # No message available - check if we should stop
                if not self._running:
                    break
                # Continue waiting (non-blocking check)
                continue
            except Exception as e:
                logger.error(f"[{self.venue_name}] Error consuming from queue: {e}")
                await asyncio.sleep(0.1)
    
    async def _reconnect(self) -> None:
        """Handle reconnection with backoff."""
        if self.max_reconnect_attempts and self._reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"[{self.venue_name}] Max reconnect attempts reached")
            self._running = False
            return
        
        self._reconnect_attempts += 1
        delay = min(self.reconnect_delay * self._reconnect_attempts, 60.0)
        logger.info(f"[{self.venue_name}] Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        
        await asyncio.sleep(delay)
        await self.connect()
    
    async def _send_subscription(self) -> None:
        """Send initial subscription message (venue-specific)."""
        message = self._build_subscription_message()
        if message:
            await self._ws.send(json.dumps(message))
            logger.info(f"[{self.venue_name}] Sent subscription message")
    
    @abstractmethod
    def _build_subscription_message(self) -> Optional[dict]:
        """Build venue-specific subscription message."""
        ...
    
    @abstractmethod
    async def _parse_message(self, message: str) -> Optional[MarketEvent]:
        """Parse venue-specific WebSocket message into MarketEvent."""
        ...

