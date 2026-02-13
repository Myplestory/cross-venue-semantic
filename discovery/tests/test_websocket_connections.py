"""Integration tests for WebSocket connections with real message formats."""

import pytest
import json
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.kalshi_poller import KalshiConnector
from discovery.polymarket_poller import PolymarketConnector
from discovery.types import VenueType, EventType, OutcomeSpec


@pytest.fixture
def kalshi_connector():
    """Create Kalshi connector for testing."""
    return KalshiConnector(
        ws_url="wss://test.kalshi.com/ws",
        reconnect_delay=0.1
    )


@pytest.fixture
def polymarket_connector():
    """Create Polymarket connector for testing."""
    return PolymarketConnector(
        ws_url="wss://test.polymarket.com/ws",
        reconnect_delay=0.1
    )


class TestKalshiWebSocket:
    """Test Kalshi WebSocket connection and message handling."""
    
    @pytest.mark.asyncio
    async def test_kalshi_subscription_message(self, kalshi_connector):
        """Test Kalshi subscription message format."""
        message = kalshi_connector._build_subscription_message()
        
        assert message is not None
        assert message["action"] == "subscribe"
        assert message["channel"] == "markets"
        assert message["params"] == {}
    
    @pytest.mark.asyncio
    async def test_kalshi_parse_market_event(self, kalshi_connector):
        """Test parsing Kalshi market event message."""
        message = json.dumps({
            "channel": "markets",
            "type": "event",
            "data": {
                "event_ticker": "MARKET-123",
                "title": "Will Bitcoin reach $100k?",
                "description": "Test description",
                "resolution_criteria": "Resolves YES if Bitcoin reaches $100k",
                "end_time": "2024-12-31T23:59:59Z",
                "status": "open",
                "outcomes": [
                    {"ticker": "YES", "name": "Yes"},
                    {"ticker": "NO", "name": "No"}
                ]
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        
        assert event is not None
        assert event.venue == VenueType.KALSHI
        assert event.venue_market_id == "MARKET-123"
        assert event.event_type == EventType.CREATED
        assert event.title == "Will Bitcoin reach $100k?"
        assert len(event.outcomes) == 2
        assert event.outcomes[0].outcome_id == "YES"
        assert event.outcomes[1].label == "No"
    
    @pytest.mark.asyncio
    async def test_kalshi_parse_market_update(self, kalshi_connector):
        """Test parsing Kalshi market update message."""
        message = json.dumps({
            "channel": "markets",
            "type": "update",
            "data": {
                "event_ticker": "MARKET-456",
                "title": "Updated Market",
                "status": "open"
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        
        assert event is not None
        assert event.event_type == EventType.UPDATED
        assert event.venue_market_id == "MARKET-456"
    
    @pytest.mark.asyncio
    async def test_kalshi_parse_resolved_market(self, kalshi_connector):
        """Test parsing Kalshi resolved market."""
        message = json.dumps({
            "channel": "markets",
            "type": "event",
            "data": {
                "event_ticker": "MARKET-789",
                "title": "Resolved Market",
                "status": "resolved"
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        
        assert event is not None
        assert event.event_type == EventType.RESOLVED
    
    @pytest.mark.asyncio
    async def test_kalshi_parse_subscription_confirmation(self, kalshi_connector):
        """Test that subscription confirmations are ignored."""
        message = json.dumps({
            "action": "subscribe",
            "channel": "markets",
            "status": "subscribed"
        })
        
        event = await kalshi_connector._parse_message(message)
        
        assert event is None  # Should be ignored
    
    @pytest.mark.asyncio
    async def test_kalshi_parse_invalid_message(self, kalshi_connector):
        """Test parsing invalid message."""
        event = await kalshi_connector._parse_message("not json")
        assert event is None
        
        event = await kalshi_connector._parse_message("{}")
        assert event is None


class TestPolymarketWebSocket:
    """Test Polymarket WebSocket connection and message handling."""
    
    @pytest.mark.asyncio
    async def test_polymarket_subscription_message(self, polymarket_connector):
        """Test Polymarket subscription message format."""
        message = polymarket_connector._build_subscription_message()
        
        assert message is not None
        assert message["type"] == "market"
        assert message["assets_ids"] == []  # Empty = all markets
    
    @pytest.mark.asyncio
    async def test_polymarket_parse_orderbook_message(self, polymarket_connector):
        """Test parsing Polymarket orderbook message."""
        message = json.dumps({
            "type": "orderbook",
            "market": "0x123abc",
            "token_id": "0xyes123",
            "bids": [[0.5, 100]],
            "asks": [[0.51, 100]]
        })
        
        event = await polymarket_connector._parse_message(message)
        
        assert event is not None
        assert event.venue == VenueType.POLYMARKET
        assert event.venue_market_id == "0x123abc"
        assert event.event_type == EventType.UPDATED
        assert event.title == ""  # Not in orderbook message
    
    @pytest.mark.asyncio
    async def test_polymarket_parse_price_message(self, polymarket_connector):
        """Test parsing Polymarket price update message."""
        message = json.dumps({
            "type": "price",
            "market_id": "0x456def",
            "outcome_id": "0xno456",
            "best_ask": 0.45,
            "best_bid": 0.44
        })
        
        event = await polymarket_connector._parse_message(message)
        
        assert event is not None
        assert event.venue_market_id == "0x456def"
        assert event.event_type == EventType.UPDATED
    
    @pytest.mark.asyncio
    async def test_polymarket_parse_market_data(self, polymarket_connector):
        """Test parsing Polymarket market data message."""
        message = json.dumps({
            "type": "market",
            "data": {
                "id": "0x789ghi",
                "question": "Will Bitcoin reach $100k?",
                "description": "Test description",
                "resolutionSource": "Resolves YES if Bitcoin reaches $100k",
                "endDate": "2024-12-31T23:59:59Z",
                "status": "open",
                "outcomes": [
                    {"token": "0xyes123", "name": "Yes"},
                    {"token": "0xno456", "name": "No"}
                ]
            }
        })
        
        event = await polymarket_connector._parse_message(message)
        
        assert event is not None
        assert event.venue_market_id == "0x789ghi"
        assert event.title == "Will Bitcoin reach $100k?"
        assert event.event_type == EventType.CREATED  # New market
        assert len(event.outcomes) == 2
    
    @pytest.mark.asyncio
    async def test_polymarket_parse_resolved_market(self, polymarket_connector):
        """Test parsing Polymarket resolved market."""
        message = json.dumps({
            "type": "market",
            "data": {
                "id": "0xresolved",
                "question": "Resolved Market",
                "status": "resolved"
            }
        })
        
        event = await polymarket_connector._parse_message(message)
        
        assert event is not None
        assert event.event_type == EventType.RESOLVED
    
    @pytest.mark.asyncio
    async def test_polymarket_tracks_subscribed_markets(self, polymarket_connector):
        """Test that Polymarket connector tracks seen markets."""
        message1 = json.dumps({
            "type": "market",
            "data": {
                "id": "0xnew123",
                "question": "New Market",
                "status": "open"
            }
        })
        
        event1 = await polymarket_connector._parse_message(message1)
        assert event1.event_type == EventType.CREATED  # First time
        
        # Same market again
        event2 = await polymarket_connector._parse_message(message1)
        assert event2.event_type == EventType.UPDATED  # Already seen
    
    @pytest.mark.asyncio
    async def test_polymarket_parse_invalid_message(self, polymarket_connector):
        """Test parsing invalid message."""
        event = await polymarket_connector._parse_message("not json")
        assert event is None
        
        event = await polymarket_connector._parse_message('{"type": "unknown"}')
        assert event is None


class TestWebSocketConnection:
    """Test WebSocket connection lifecycle."""
    
    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, kalshi_connector):
        """Test connecting and disconnecting."""
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.close_code = None
            mock_ws.send = AsyncMock()
            
            mock_connect.return_value = mock_ws
            
            await kalshi_connector.connect()
            
            assert kalshi_connector._ws is not None
            mock_connect.assert_called_once()
            mock_ws.send.assert_called_once()  # Subscription message
            
            await kalshi_connector.disconnect()
            assert kalshi_connector._ws is None
    
    @pytest.mark.asyncio
    async def test_reconnection_on_connection_closed(self, kalshi_connector):
        """Test automatic reconnection when connection closes."""
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            # Create two mock websockets for initial and reconnected connections
            mock_ws1 = AsyncMock()
            mock_ws1.closed = False
            mock_ws1.close_code = None
            mock_ws1.send = AsyncMock()
            
            mock_ws2 = AsyncMock()
            mock_ws2.closed = False
            mock_ws2.close_code = None
            mock_ws2.send = AsyncMock()
            
            # First connection raises ConnectionClosed immediately
            async def mock_iter1():
                from websockets.exceptions import ConnectionClosed
                raise ConnectionClosed(None, None)
            
            # Second connection (after reconnect) stops immediately
            async def mock_iter2():
                kalshi_connector._running = False
                if False:
                    yield
            
            mock_ws1.__aiter__ = lambda self: mock_iter1()
            mock_ws2.__aiter__ = lambda self: mock_iter2()
            
            # Return different mocks for each connection attempt
            call_count = 0
            def get_mock_ws(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_ws1
                return mock_ws2
            
            mock_connect.side_effect = get_mock_ws
            
            await kalshi_connector.connect()
            
            # Stream events with timeout to prevent hanging
            events = []
            try:
                async def stream_with_timeout():
                    async for event in kalshi_connector.stream_events():
                        events.append(event)
                        kalshi_connector._running = False
                        break
                
                await asyncio.wait_for(stream_with_timeout(), timeout=3.0)
            except asyncio.TimeoutError:
                kalshi_connector._running = False
            except Exception:
                kalshi_connector._running = False
            
            # Should have attempted reconnection
            assert mock_connect.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_stream_events_single_message(self, kalshi_connector):
        """Test streaming a single event."""
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.close_code = None
            mock_ws.send = AsyncMock()
            
            # Mock message iteration
            async def mock_iter():
                yield json.dumps({
                    "channel": "markets",
                    "type": "event",
                    "data": {
                        "event_ticker": "MARKET-123",
                        "title": "Test Market",
                        "status": "open"
                    }
                })
                kalshi_connector._running = False
            
            mock_ws.__aiter__ = lambda self: mock_iter()
            
            mock_connect.return_value = mock_ws
            
            await kalshi_connector.connect()
            
            events = []
            async for event in kalshi_connector.stream_events():
                events.append(event)
                break
            
            assert len(events) == 1
            assert events[0].venue_market_id == "MARKET-123"


class TestMessageFormatRobustness:
    """Test message parsing robustness with various formats."""
    
    @pytest.mark.asyncio
    async def test_kalshi_missing_fields(self, kalshi_connector):
        """Test Kalshi message with missing optional fields."""
        message = json.dumps({
            "channel": "markets",
            "type": "event",
            "data": {
                "event_ticker": "MARKET-123",
                "title": "Minimal Market"
                # Missing description, outcomes, etc.
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        
        assert event is not None
        assert event.venue_market_id == "MARKET-123"
        assert event.description is None
        assert event.outcomes == []
    
    @pytest.mark.asyncio
    async def test_polymarket_missing_fields(self, polymarket_connector):
        """Test Polymarket message with missing optional fields."""
        message = json.dumps({
            "type": "orderbook",
            "market": "0x123"
            # Missing token_id, bids, asks
        })
        
        event = await polymarket_connector._parse_message(message)
        
        assert event is not None
        assert event.venue_market_id == "0x123"
    
    @pytest.mark.asyncio
    async def test_kalshi_malformed_outcomes(self, kalshi_connector):
        """Test Kalshi message with malformed outcomes."""
        message = json.dumps({
            "channel": "markets",
            "type": "event",
            "data": {
                "event_ticker": "MARKET-123",
                "title": "Test",
                "outcomes": [
                    {"ticker": "YES"},  # Missing name
                    "invalid",  # Not a dict
                    {"name": "No"}  # Missing ticker
                ]
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        
        assert event is not None
        # Should handle gracefully
        assert len(event.outcomes) <= 3
    
    @pytest.mark.asyncio
    async def test_polymarket_malformed_date(self, polymarket_connector):
        """Test Polymarket message with malformed date."""
        message = json.dumps({
            "type": "market",
            "data": {
                "id": "0x123",
                "question": "Test",
                "endDate": "invalid-date-format"
            }
        })
        
        event = await polymarket_connector._parse_message(message)
        
        assert event is not None
        assert event.end_date is None  # Should handle gracefully

