"""Tests for Polymarket connector."""

import pytest
import json
from unittest.mock import AsyncMock, patch

from ..polymarket_poller import PolymarketConnector
from ..types import VenueType, EventType, OutcomeSpec


@pytest.fixture
def polymarket_connector():
    """Create Polymarket connector instance."""
    return PolymarketConnector(
        ws_url="wss://test.polymarket.com/ws",
        reconnect_delay=0.1
    )


@pytest.fixture
def sample_polymarket_message():
    """Sample Polymarket WebSocket message."""
    return json.dumps({
        "type": "market",
        "data": {
            "id": "0x123abc",
            "question": "Will Bitcoin reach $100k?",
            "description": "Test description",
            "resolutionSource": "Resolves YES if Bitcoin reaches $100k",
            "endDate": "2024-12-31T23:59:59Z",
            "status": "open",
            "outcomes": [
                {"token": "0xabc123", "name": "Yes"},
                {"token": "0xdef456", "name": "No"}
            ]
        }
    })


@pytest.mark.asyncio
async def test_polymarket_parse_message_created(polymarket_connector, sample_polymarket_message):
    """Test parsing Polymarket market_created message."""
    event = await polymarket_connector._parse_message(sample_polymarket_message)
    
    assert event is not None
    assert event.venue == VenueType.POLYMARKET
    assert event.venue_market_id == "0x123abc"
    assert event.event_type == EventType.CREATED
    assert event.title == "Will Bitcoin reach $100k?"
    assert event.description == "Test description"
    assert event.resolution_criteria == "Resolves YES if Bitcoin reaches $100k"
    assert len(event.outcomes) == 2
    assert event.outcomes[0].outcome_id == "0xabc123"
    assert event.outcomes[0].label == "Yes"
    assert event.outcomes[1].outcome_id == "0xdef456"
    assert event.outcomes[1].label == "No"
    assert event.end_date is not None


@pytest.mark.asyncio
async def test_polymarket_parse_message_updated(polymarket_connector):
    """Test parsing Polymarket market_updated message."""
    # First message creates the market
    message1 = json.dumps({
        "type": "market",
        "data": {
            "id": "0x456def",
            "question": "Updated Market",
            "status": "open"
        }
    })
    await polymarket_connector._parse_message(message1)
    
    # Second message updates it
    message2 = json.dumps({
        "type": "orderbook",
        "market": "0x456def",
        "token_id": "0xtoken"
    })
    
    event = await polymarket_connector._parse_message(message2)
    
    assert event is not None
    assert event.event_type == EventType.UPDATED
    assert event.venue_market_id == "0x456def"


@pytest.mark.asyncio
async def test_polymarket_parse_message_resolved(polymarket_connector):
    """Test parsing Polymarket market_resolved message."""
    message = json.dumps({
        "type": "market",
        "data": {
            "id": "0x789ghi",
            "question": "Resolved Market",
            "status": "resolved",
            "outcomes": []
        }
    })
    
    event = await polymarket_connector._parse_message(message)
    
    assert event is not None
    assert event.event_type == EventType.RESOLVED


@pytest.mark.asyncio
async def test_polymarket_parse_message_unknown_type(polymarket_connector):
    """Test parsing unknown message type."""
    message = json.dumps({
        "type": "unknown_type",
        "data": {}
    })
    
    event = await polymarket_connector._parse_message(message)
    
    assert event is None


@pytest.mark.asyncio
async def test_polymarket_parse_message_invalid_json(polymarket_connector):
    """Test parsing invalid JSON."""
    event = await polymarket_connector._parse_message("not json")
    
    assert event is None


@pytest.mark.asyncio
async def test_polymarket_build_subscription_message(polymarket_connector):
    """Test building subscription message."""
    message = polymarket_connector._build_subscription_message()
    
    assert message is not None
    assert message["type"] == "market"
    assert message["assets_ids"] == []  # Empty = all markets


@pytest.mark.asyncio
async def test_polymarket_connect_and_disconnect(polymarket_connector):
    """Test connecting and disconnecting."""
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        
        async def mock_connect_func(*args, **kwargs):
            return mock_ws
        mock_connect.side_effect = mock_connect_func
        
        await polymarket_connector.connect()
        
        assert polymarket_connector._ws is not None
        mock_connect.assert_called_once()
        
        await polymarket_connector.disconnect()
        
        assert polymarket_connector._ws is None

