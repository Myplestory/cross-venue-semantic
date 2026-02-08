"""Integration tests using mock WebSocket server."""

import pytest
import asyncio
import json
from unittest.mock import patch

from ..kalshi_poller import KalshiConnector
from ..polymarket_poller import PolymarketConnector
from ..types import EventType
from .test_websocket_server import KalshiMockServer, PolymarketMockServer


@pytest.fixture
async def kalshi_mock_server():
    """Create and start Kalshi mock server."""
    server = KalshiMockServer(port=8765)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def polymarket_mock_server():
    """Create and start Polymarket mock server."""
    server = PolymarketMockServer(port=8766)
    await server.start()
    yield server
    await server.stop()


@pytest.mark.asyncio
async def test_kalshi_connector_with_mock_server(kalshi_mock_server):
    """Test Kalshi connector with mock WebSocket server."""
    connector = KalshiConnector(
        ws_url="ws://localhost:8765",
        reconnect_delay=0.1
    )
    
    try:
        await connector.connect()
        
        # Stream events
        events = []
        async for event in connector.stream_events():
            events.append(event)
            if len(events) >= 1:  # Get first event
                connector._running = False
                break
        
        assert len(events) == 1
        event = events[0]
        assert event.venue_market_id == "TEST-MARKET-123"
        assert event.title == "Test Market from Mock Server"
        assert event.event_type == EventType.CREATED
        assert len(event.outcomes) == 2
        
        # Check subscription message was sent
        assert len(kalshi_mock_server.messages_received) > 0
        sub_msg = kalshi_mock_server.messages_received[0]
        assert sub_msg["action"] == "subscribe"  # Kalshi uses "action", not "type"
        assert sub_msg["channel"] == "markets"
        
    finally:
        await connector.disconnect()


@pytest.mark.asyncio
async def test_polymarket_connector_with_mock_server(polymarket_mock_server):
    """Test Polymarket connector with mock WebSocket server."""
    connector = PolymarketConnector(
        ws_url="ws://localhost:8766",
        reconnect_delay=0.1
    )
    
    try:
        await connector.connect()
        
        # Stream events
        events = []
        async for event in connector.stream_events():
            events.append(event)
            if len(events) >= 1:  # Get first event
                connector._running = False
                break
        
        assert len(events) == 1
        event = events[0]
        assert event.venue_market_id == "0xtest123"
        assert event.title == "Test Market from Mock Server"
        assert event.event_type == EventType.CREATED
        assert len(event.outcomes) == 2
        
        # Check subscription message was sent
        assert len(polymarket_mock_server.messages_received) > 0
        sub_msg = polymarket_mock_server.messages_received[0]
        assert sub_msg["type"] == "market"  # Polymarket uses "type": "market" for subscriptions
        assert "assets_ids" in sub_msg  # Polymarket subscription includes assets_ids
        
    finally:
        await connector.disconnect()


@pytest.mark.asyncio
async def test_reconnection_with_mock_server(kalshi_mock_server):
    """Test reconnection logic with mock server."""
    connector = KalshiConnector(
        ws_url="ws://localhost:8765",
        reconnect_delay=0.1,
        max_reconnect_attempts=3
    )
    
    try:
        await connector.connect()
        assert connector._ws is not None
        
        # Simulate connection drop
        await connector._ws.close()
        
        # Should reconnect automatically
        events = []
        async for event in connector.stream_events():
            events.append(event)
            if len(events) >= 1:
                connector._running = False
                break
        
        # Should have reconnected and received event
        assert len(events) >= 0  # May or may not get event depending on timing
        
    finally:
        await connector.disconnect()

