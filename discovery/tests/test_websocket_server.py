"""
Mock WebSocket server for testing venue connectors.

Provides a test WebSocket server that can simulate venue responses.
"""

import asyncio
import json
import logging
from typing import Optional, Callable
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class MockWebSocketServer:
    """Mock WebSocket server for testing."""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.server: Optional[websockets.server.Serve] = None
        self.clients: list[WebSocketServerProtocol] = []
        self.messages_received: list[dict] = []
        self.message_handler: Optional[Callable] = None
    
    async def start(self):
        """Start the mock WebSocket server."""
        # Handler function - websockets.serve passes (websocket, path)
        # Make path optional to handle different websockets library versions
        async def handler(websocket, path=None):
            await self._handle_client(websocket, path or "/")
        
        self.server = await websockets.serve(
            handler,
            "localhost",
            self.port
        )
        logger.info(f"Mock WebSocket server started on ws://localhost:{self.port}")
    
    async def stop(self):
        """Stop the mock WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Mock WebSocket server stopped")
    
    async def _handle_client(self, ws: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket client connection."""
        self.clients.append(ws)
        logger.info(f"Client connected: {path}")
        
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    self.messages_received.append(data)
                    
                    # Call custom handler if set
                    if self.message_handler:
                        response = await self.message_handler(data)
                        if response:
                            await ws.send(json.dumps(response))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            if ws in self.clients:
                self.clients.remove(ws)
    
    async def send_to_all(self, message: dict):
        """Send a message to all connected clients."""
        message_str = json.dumps(message)
        disconnected = []
        
        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
        
        for client in disconnected:
            if client in self.clients:
                self.clients.remove(client)
    
    def set_message_handler(self, handler: Callable):
        """Set a custom message handler."""
        self.message_handler = handler


class KalshiMockServer(MockWebSocketServer):
    """Mock server that simulates Kalshi WebSocket responses."""
    
    async def start(self):
        """Start the mock WebSocket server."""
        # Handler function - websockets.serve passes (websocket, path)
        # Make path optional to handle different websockets library versions
        async def handler(websocket, path=None):
            await self._handle_client(websocket, path or "/")
        
        self.server = await websockets.serve(
            handler,
            "localhost",
            self.port
        )
        logger.info(f"Mock WebSocket server started on ws://localhost:{self.port}")
    
    async def _handle_client(self, ws: WebSocketServerProtocol, path: str):
        """Handle Kalshi client with subscription acknowledgment."""
        self.clients.append(ws)
        logger.info(f"Kalshi client connected: {path}")
        
        # Send subscription acknowledgment
        await ws.send(json.dumps({
            "type": "subscription_confirmed",
            "channel": "markets"
        }))
        
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    self.messages_received.append(data)
                    
                    # Handle subscription request
                    if data.get("action") == "subscribe" and data.get("channel") == "markets":
                        # Send a test market event
                        await ws.send(json.dumps({
                            "channel": "markets",
                            "type": "event",
                            "data": {
                                "event_ticker": "TEST-MARKET-123",
                                "title": "Test Market from Mock Server",
                                "description": "This is a test market",
                                "resolution_criteria": "Resolves YES if test passes",
                                "end_time": "2024-12-31T23:59:59Z",
                                "status": "open",
                                "outcomes": [
                                    {"ticker": "YES", "name": "Yes"},
                                    {"ticker": "NO", "name": "No"}
                                ]
                            }
                        }))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Kalshi client disconnected")
        finally:
            if ws in self.clients:
                self.clients.remove(ws)


class PolymarketMockServer(MockWebSocketServer):
    """Mock server that simulates Polymarket WebSocket responses."""
    
    async def start(self):
        """Start the mock WebSocket server."""
        # Handler function - websockets.serve passes (websocket, path)
        # Make path optional to handle different websockets library versions
        async def handler(websocket, path=None):
            await self._handle_client(websocket, path or "/")
        
        self.server = await websockets.serve(
            handler,
            "localhost",
            self.port
        )
        logger.info(f"Mock WebSocket server started on ws://localhost:{self.port}")
    
    async def _handle_client(self, ws: WebSocketServerProtocol, path: str):
        """Handle Polymarket client with subscription acknowledgment."""
        self.clients.append(ws)
        logger.info(f"Polymarket client connected: {path}")
        
        # Send subscription acknowledgment
        await ws.send(json.dumps({
            "type": "subscription_confirmed",
            "channel": "markets"
        }))
        
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    self.messages_received.append(data)
                    
                    # Handle subscription request
                    if data.get("type") == "market" and data.get("assets_ids") == []:
                        # Send a test market event
                        await ws.send(json.dumps({
                            "type": "market",
                            "data": {
                                "id": "0xtest123",
                                "question": "Test Market from Mock Server",
                                "description": "This is a test market",
                                "resolutionSource": "Resolves YES if test passes",
                                "endDate": "2024-12-31T23:59:59Z",
                                "status": "open",
                                "outcomes": [
                                    {"token": "0xyes123", "name": "Yes"},
                                    {"token": "0xno456", "name": "No"}
                                ]
                            }
                        }))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Polymarket client disconnected")
        finally:
            if ws in self.clients:
                self.clients.remove(ws)

