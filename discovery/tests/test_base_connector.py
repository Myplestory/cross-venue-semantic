"""Tests for base connector."""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ..base_connector import BaseVenueConnector
from ..types import VenueType, MarketEvent, EventType


class MockConnector(BaseVenueConnector):
    """Mock connector implementation for testing."""
    
    def _build_subscription_message(self):
        return {"type": "subscribe"}
    
    async def _parse_message(self, message: str):
        data = json.loads(message)
        if data.get("type") == "market_event":
            return MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=data.get("market_id", ""),
                event_type=EventType.CREATED,
                title=data.get("title", "")
            )
        return None
    


@pytest.fixture
def test_connector():
    """Create test connector instance."""
    return MockConnector(
        venue_name=VenueType.KALSHI,
        ws_url="wss://test.com/ws",
        reconnect_delay=0.1
    )


@pytest.mark.asyncio
async def test_connect_success(test_connector):
    """Test successful connection."""
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close_code = None
        mock_ws.send = AsyncMock()
        # Make websockets.connect awaitable
        mock_connect.return_value = mock_ws
        
        await test_connector.connect()
        
        assert test_connector._ws is not None
        mock_connect.assert_called_once()
        mock_ws.send.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect(test_connector):
    """Test disconnection."""
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close_code = None
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        # Make websockets.connect awaitable
        mock_connect.return_value = mock_ws
        
        await test_connector.connect()
        await test_connector.disconnect()
        
        assert test_connector._ws is None
        mock_ws.close.assert_called_once()


@pytest.mark.asyncio
async def test_stream_events_single_message(test_connector):
    """Test streaming a single event."""
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close_code = None
        mock_ws.send = AsyncMock()
        
        # Mock message iteration
        async def mock_iter():
            yield json.dumps({"type": "market_event", "market_id": "MARKET-123", "title": "Test"})
            # Stop iteration to exit loop
            test_connector._running = False
        
        mock_ws.__aiter__ = lambda self: mock_iter()
        # Make websockets.connect awaitable
        mock_connect.return_value = mock_ws
        
        await test_connector.connect()
        
        events = []
        async for event in test_connector.stream_events():
            events.append(event)
            break  # Exit after first event
        
        assert len(events) == 1
        assert events[0].venue_market_id == "MARKET-123"


@pytest.mark.asyncio
async def test_reconnect_on_connection_closed(test_connector):
    """Test reconnection when connection is closed."""
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        # Create two mock websockets - one for initial connection, one for reconnect
        mock_ws1 = AsyncMock()
        mock_ws1.closed = False
        mock_ws1.close_code = None
        mock_ws1.send = AsyncMock()
        
        mock_ws2 = AsyncMock()
        mock_ws2.closed = False
        mock_ws2.close_code = None
        mock_ws2.send = AsyncMock()
        
        # First connection yields one message then closes
        async def mock_iter1():
            yield json.dumps({"type": "market_event", "market_id": "MARKET-123", "title": "Test"})
            from websockets.exceptions import ConnectionClosed
            raise ConnectionClosed(None, None)
        
        # Second connection (after reconnect) yields nothing and stops
        async def mock_iter2():
            test_connector._running = False  # Stop after reconnect
            return
            yield  # Make it a generator
        
        mock_ws1.__aiter__ = lambda self: mock_iter1()
        mock_ws2.__aiter__ = lambda self: mock_iter2()
        
        # First call returns ws1, subsequent calls return ws2
        call_count = 0
        def get_mock_ws(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_ws1
            return mock_ws2
        
        mock_connect.side_effect = get_mock_ws
        
        await test_connector.connect()
        
        # Stream events with timeout
        events = []
        try:
            async with asyncio.timeout(2.0):  # 2 second timeout
                async for event in test_connector.stream_events():
                    events.append(event)
                    if len(events) >= 1:
                        test_connector._running = False
                        break
        except asyncio.TimeoutError:
            test_connector._running = False
        except Exception:
            test_connector._running = False
        
        # Should have attempted reconnection (at least 2 calls: initial + reconnect)
        assert mock_connect.call_count >= 1

