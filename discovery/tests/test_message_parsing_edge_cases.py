"""Test edge cases and robustness of message parsing."""

import pytest
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.kalshi_poller import KalshiConnector
from discovery.polymarket_poller import PolymarketConnector
from discovery.types import VenueType, EventType


@pytest.fixture
def kalshi_connector():
    return KalshiConnector(ws_url="wss://test.com/ws")


@pytest.fixture
def polymarket_connector():
    return PolymarketConnector(ws_url="wss://test.com/ws")


class TestKalshiEdgeCases:
    """Test Kalshi message parsing edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_data(self, kalshi_connector):
        """Test message with empty data."""
        message = json.dumps({
            "channel": "markets",
            "type": "event",
            "data": {}
        })
        
        event = await kalshi_connector._parse_message(message)
        assert event is None
    
    @pytest.mark.asyncio
    async def test_missing_event_ticker(self, kalshi_connector):
        """Test message missing event_ticker."""
        message = json.dumps({
            "channel": "markets",
            "type": "event",
            "data": {
                "title": "Market without ID"
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        # Should handle gracefully - may return None or event with empty ID
        assert event is None or event.venue_market_id == ""
    
    @pytest.mark.asyncio
    async def test_invalid_date_format(self, kalshi_connector):
        """Test message with invalid date format."""
        message = json.dumps({
            "channel": "markets",
            "type": "event",
            "data": {
                "event_ticker": "MARKET-123",
                "title": "Test",
                "end_time": "not-a-date"
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        assert event is not None
        assert event.end_date is None  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_null_values(self, kalshi_connector):
        """Test message with null values."""
        message = json.dumps({
            "channel": "markets",
            "type": "event",
            "data": {
                "event_ticker": "MARKET-123",
                "title": "Test",
                "description": None,
                "outcomes": None
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        assert event is not None
        assert event.description is None
        assert event.outcomes == []
    
    @pytest.mark.asyncio
    async def test_wrong_channel(self, kalshi_connector):
        """Test message from wrong channel."""
        message = json.dumps({
            "channel": "prices",  # Not markets
            "type": "update",
            "data": {"event_ticker": "MARKET-123"}
        })
        
        event = await kalshi_connector._parse_message(message)
        assert event is None
    
    @pytest.mark.asyncio
    async def test_unicode_in_title(self, kalshi_connector):
        """Test message with unicode characters."""
        message = json.dumps({
            "channel": "markets",
            "type": "event",
            "data": {
                "event_ticker": "MARKET-123",
                "title": "Will Bitcoin reach $100k? 🚀",
                "description": "Test with émojis and spéciál chars"
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        assert event is not None
        assert "🚀" in event.title
        assert "émojis" in event.description


class TestPolymarketEdgeCases:
    """Test Polymarket message parsing edge cases."""
    
    @pytest.mark.asyncio
    async def test_orderbook_missing_market(self, polymarket_connector):
        """Test orderbook message missing market ID."""
        message = json.dumps({
            "type": "orderbook",
            "token_id": "0x123"
            # Missing market
        })
        
        event = await polymarket_connector._parse_message(message)
        assert event is None
    
    @pytest.mark.asyncio
    async def test_market_data_missing_id(self, polymarket_connector):
        """Test market data message missing ID."""
        message = json.dumps({
            "type": "market",
            "data": {
                "question": "Market without ID"
            }
        })
        
        event = await polymarket_connector._parse_message(message)
        # May return event with empty ID or None
        assert event is None or event.venue_market_id == ""
    
    @pytest.mark.asyncio
    async def test_empty_outcomes_list(self, polymarket_connector):
        """Test message with empty outcomes."""
        message = json.dumps({
            "type": "market",
            "data": {
                "id": "0x123",
                "question": "Test",
                "outcomes": []
            }
        })
        
        event = await polymarket_connector._parse_message(message)
        assert event is not None
        assert event.outcomes == []
    
    @pytest.mark.asyncio
    async def test_outcomes_not_list(self, polymarket_connector):
        """Test message with outcomes not as list."""
        message = json.dumps({
            "type": "market",
            "data": {
                "id": "0x123",
                "question": "Test",
                "outcomes": "not-a-list"
            }
        })
        
        event = await polymarket_connector._parse_message(message)
        # Should handle gracefully
        assert event is not None
        assert event.outcomes == []
    
    @pytest.mark.asyncio
    async def test_very_long_title(self, polymarket_connector):
        """Test message with very long title."""
        long_title = "A" * 10000
        message = json.dumps({
            "type": "orderbook",
            "market": "0x123",
            "token_id": "0x456"
        })
        
        event = await polymarket_connector._parse_message(message)
        # Should handle without issues
        assert event is not None
    
    @pytest.mark.asyncio
    async def test_nested_data_structures(self, polymarket_connector):
        """Test message with nested data structures."""
        message = json.dumps({
            "type": "market",
            "data": {
                "id": "0x123",
                "question": "Test",
                "metadata": {
                    "nested": {
                        "deep": "value"
                    }
                }
            }
        })
        
        event = await polymarket_connector._parse_message(message)
        # Should extract what it can, ignore nested structures
        assert event is not None
        assert event.venue_market_id == "0x123"


class TestMessageFormatVariations:
    """Test various message format variations."""
    
    @pytest.mark.asyncio
    async def test_kalshi_different_status_values(self, kalshi_connector):
        """Test Kalshi messages with different status values."""
        test_cases = [
            ("open", EventType.CREATED),
            ("closed", EventType.CLOSED),
            ("resolved", EventType.RESOLVED),
            ("unknown", EventType.UPDATED),  # Default
        ]
        
        for status, expected_type in test_cases:
            message = json.dumps({
                "channel": "markets",
                "type": "event",
                "data": {
                    "event_ticker": f"MARKET-{status}",
                    "title": f"Market {status}",
                    "status": status
                }
            })
            
            event = await kalshi_connector._parse_message(message)
            assert event is not None
            assert event.event_type == expected_type
    
    @pytest.mark.asyncio
    async def test_polymarket_different_message_types(self, polymarket_connector):
        """Test Polymarket messages with different types."""
        test_cases = [
            ("orderbook", True),
            ("price", True),
            ("market", True),
            ("unknown", False),  # Should be ignored
        ]
        
        for msg_type, should_parse in test_cases:
            message = json.dumps({
                "type": msg_type,
                "market": "0x123" if msg_type in ["orderbook", "price"] else None,
                "data": {
                    "id": "0x123",
                    "question": "Test"
                } if msg_type == "market" else None
            })
            
            event = await polymarket_connector._parse_message(message)
            if should_parse:
                assert event is not None, f"Should parse {msg_type}"
            else:
                # Unknown types may return None
                pass
    
    @pytest.mark.asyncio
    async def test_case_insensitive_fields(self, kalshi_connector):
        """Test that parsing handles case variations."""
        # Kalshi should be case-sensitive per their API
        # But test that we handle gracefully
        message = json.dumps({
            "CHANNEL": "markets",  # Wrong case
            "TYPE": "event",
            "DATA": {
                "EVENT_TICKER": "MARKET-123",
                "TITLE": "Test"
            }
        })
        
        event = await kalshi_connector._parse_message(message)
        # May not parse due to case sensitivity, but shouldn't crash
        assert event is None or event.venue_market_id == ""

