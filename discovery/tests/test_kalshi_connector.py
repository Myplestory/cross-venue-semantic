"""Tests for Kalshi connector."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ..kalshi_poller import KalshiConnector
from ..types import VenueType, EventType, OutcomeSpec


@pytest.fixture
def kalshi_connector():
    """Create Kalshi connector instance."""
    return KalshiConnector(
        ws_url="wss://test.kalshi.com/ws",
        reconnect_delay=0.1
    )


@pytest.fixture
def sample_kalshi_message():
    """Sample Kalshi WebSocket message."""
    return json.dumps({
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


@pytest.mark.asyncio
async def test_kalshi_parse_message_created(kalshi_connector, sample_kalshi_message):
    """Test parsing Kalshi market_created message."""
    event = await kalshi_connector._parse_message(sample_kalshi_message)
    
    assert event is not None
    assert event.venue == VenueType.KALSHI
    assert event.venue_market_id == "MARKET-123"
    assert event.event_type == EventType.CREATED
    assert event.title == "Will Bitcoin reach $100k?"
    assert event.description == "Test description"
    assert event.resolution_criteria == "Resolves YES if Bitcoin reaches $100k"
    assert len(event.outcomes) == 2
    assert event.outcomes[0].outcome_id == "YES"
    assert event.outcomes[0].label == "Yes"
    assert event.outcomes[1].outcome_id == "NO"
    assert event.outcomes[1].label == "No"
    assert event.end_date is not None


@pytest.mark.asyncio
async def test_kalshi_parse_message_updated(kalshi_connector):
    """Test parsing Kalshi market_updated message."""
    message = json.dumps({
        "channel": "markets",
        "type": "update",
        "data": {
            "event_ticker": "MARKET-456",
            "title": "Updated Market",
            "status": "open",
            "outcomes": []
        }
    })
    
    event = await kalshi_connector._parse_message(message)
    
    assert event is not None
    assert event.event_type == EventType.UPDATED
    assert event.venue_market_id == "MARKET-456"


@pytest.mark.asyncio
async def test_kalshi_parse_message_resolved(kalshi_connector):
    """Test parsing Kalshi market_resolved message."""
    message = json.dumps({
        "channel": "markets",
        "type": "event",
        "data": {
            "event_ticker": "MARKET-789",
            "title": "Resolved Market",
            "status": "resolved",
            "outcomes": []
        }
    })
    
    event = await kalshi_connector._parse_message(message)
    
    assert event is not None
    assert event.event_type == EventType.RESOLVED


@pytest.mark.asyncio
async def test_kalshi_parse_message_unknown_type(kalshi_connector):
    """Test parsing unknown message type."""
    message = json.dumps({
        "type": "unknown_type",
        "data": {}
    })
    
    event = await kalshi_connector._parse_message(message)
    
    assert event is None


@pytest.mark.asyncio
async def test_kalshi_parse_message_invalid_json(kalshi_connector):
    """Test parsing invalid JSON."""
    event = await kalshi_connector._parse_message("not json")
    
    assert event is None


@pytest.mark.asyncio
async def test_kalshi_build_subscription_message(kalshi_connector):
    """Test building subscription message."""
    message = kalshi_connector._build_subscription_message()
    
    assert message is not None
    assert message["action"] == "subscribe"
    assert message["channel"] == "markets"
    assert message["params"] == {}


@pytest.mark.asyncio
async def test_kalshi_connect_and_disconnect(kalshi_connector):
    """Test connecting and disconnecting."""
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        
        async def mock_connect_func(*args, **kwargs):
            return mock_ws
        mock_connect.side_effect = mock_connect_func
        
        await kalshi_connector.connect()
        
        assert kalshi_connector._ws is not None
        mock_connect.assert_called_once()
        
        await kalshi_connector.disconnect()
        
        assert kalshi_connector._ws is None

