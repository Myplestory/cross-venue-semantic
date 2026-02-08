"""Integration tests for full market discovery flow."""

import pytest
import json
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.venue_factory import create_connector
from discovery.types import VenueType, EventType
from discovery.dedup import MarketDeduplicator


@pytest.fixture
def deduplicator():
    """Create deduplicator for testing."""
    return MarketDeduplicator(ttl_seconds=None)


class TestFullDiscoveryFlow:
    """Test complete market discovery flow with multiple venues."""
    
    @pytest.mark.asyncio
    async def test_single_venue_discovery_flow(self, deduplicator):
        """Test complete flow: connect -> stream -> deduplicate -> process."""
        connector = create_connector(VenueType.KALSHI, reconnect_delay=0.1)
        
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.send = AsyncMock()
            
            messages = [
                json.dumps({
                    "channel": "markets",
                    "type": "event",
                    "data": {
                        "event_ticker": "MARKET-1",
                        "title": "Market 1",
                        "status": "open"
                    }
                }),
                json.dumps({
                    "channel": "markets",
                    "type": "event",
                    "data": {
                        "event_ticker": "MARKET-2",
                        "title": "Market 2",
                        "status": "open"
                    }
                }),
            ]
            
            async def mock_iter():
                for msg in messages:
                    yield msg
                connector._running = False
            
            mock_ws.__aiter__ = lambda self: mock_iter()
            
            mock_connect.return_value = mock_ws
            
            await connector.connect()
            
            # Full flow: stream -> deduplicate -> collect
            discovered_markets = []
            async for event in connector.stream_events():
                if deduplicator.is_duplicate(event):
                    continue
                
                discovered_markets.append(event)
            
            assert len(discovered_markets) == 2
            assert discovered_markets[0].venue_market_id == "MARKET-1"
            assert discovered_markets[1].venue_market_id == "MARKET-2"
            
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_deduplication_in_flow(self, deduplicator):
        """Test that deduplication works in full flow."""
        connector = create_connector(VenueType.KALSHI, reconnect_delay=0.1)
        
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.send = AsyncMock()
            
            # Send same market twice
            message = json.dumps({
                "channel": "markets",
                "type": "event",
                "data": {
                    "event_ticker": "MARKET-DUP",
                    "title": "Duplicate Market",
                    "status": "open"
                }
            })
            
            async def mock_iter():
                yield message
                yield message  # Duplicate
                connector._running = False
            
            mock_ws.__aiter__ = lambda self: mock_iter()
            
            mock_connect.return_value = mock_ws
            
            await connector.connect()
            
            discovered = []
            async for event in connector.stream_events():
                if deduplicator.is_duplicate(event):
                    continue
                discovered.append(event)
            
            # Should only get one (deduplicated)
            assert len(discovered) == 1
            assert discovered[0].venue_market_id == "MARKET-DUP"
            
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_venues_parallel(self, deduplicator):
        """Test streaming from multiple venues in parallel."""
        connectors = {
            VenueType.KALSHI: create_connector(VenueType.KALSHI, reconnect_delay=0.1),
            VenueType.POLYMARKET: create_connector(VenueType.POLYMARKET, reconnect_delay=0.1),
        }
        
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.send = AsyncMock()
            
            async def kalshi_iter():
                yield json.dumps({
                    "channel": "markets",
                    "type": "event",
                    "data": {
                        "event_ticker": "KALSHI-1",
                        "title": "Kalshi Market",
                        "status": "open"
                    }
                })
                connectors[VenueType.KALSHI]._running = False
            
            async def polymarket_iter():
                yield json.dumps({
                    "type": "orderbook",
                    "market": "0xPOLY1",
                    "token_id": "0xtoken1"
                })
                connectors[VenueType.POLYMARKET]._running = False
            
            mock_ws_kalshi = AsyncMock()
            mock_ws_kalshi.closed = False
            mock_ws_kalshi.send = AsyncMock()
            mock_ws_kalshi.__aiter__ = lambda self: kalshi_iter()
            
            mock_ws_poly = AsyncMock()
            mock_ws_poly.closed = False
            mock_ws_poly.send = AsyncMock()
            mock_ws_poly.__aiter__ = lambda self: polymarket_iter()
            
            # For AsyncMock with side_effect, the function should return the value directly
            # AsyncMock will handle the awaiting
            def connect_side_effect(*args, **kwargs):
                url = args[0] if args else kwargs.get("uri", "") or kwargs.get("uri", "")
                url_str = str(url)
                if "kalshi" in url_str:
                    return mock_ws_kalshi
                return mock_ws_poly
            
            mock_connect.side_effect = connect_side_effect
            
            # Connect both
            for connector in connectors.values():
                await connector.connect()
            
            # Stream from both
            all_events = []
            
            async def stream_venue(venue_type, connector):
                async for event in connector.stream_events():
                    if deduplicator.is_duplicate(event):
                        continue
                    all_events.append((venue_type, event))
            
            tasks = [
                asyncio.create_task(stream_venue(VenueType.KALSHI, connectors[VenueType.KALSHI])),
                asyncio.create_task(stream_venue(VenueType.POLYMARKET, connectors[VenueType.POLYMARKET])),
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should have events from both venues
            assert len(all_events) >= 2
            venues_seen = {venue for venue, _ in all_events}
            assert VenueType.KALSHI in venues_seen
            assert VenueType.POLYMARKET in venues_seen
            
            # Cleanup
            for connector in connectors.values():
                await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_flow(self, deduplicator):
        """Test error handling during streaming."""
        connector = create_connector(VenueType.KALSHI, reconnect_delay=0.1)
        
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.send = AsyncMock()
            
            async def mock_iter():
                yield json.dumps({
                    "channel": "markets",
                    "type": "event",
                    "data": {"event_ticker": "MARKET-1", "title": "Good"}
                })
                yield "invalid json"  # Error
                yield json.dumps({
                    "channel": "markets",
                    "type": "event",
                    "data": {"event_ticker": "MARKET-2", "title": "Good"}
                })
                connector._running = False
            
            mock_ws.__aiter__ = lambda self: mock_iter()
            
            mock_connect.return_value = mock_ws
            
            await connector.connect()
            
            # Should handle error gracefully and continue
            events = []
            async for event in connector.stream_events():
                if deduplicator.is_duplicate(event):
                    continue
                events.append(event)
            
            # Should get both valid events despite error
            assert len(events) == 2
            
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_event_type_detection(self, deduplicator):
        """Test that event types are correctly detected."""
        connector = create_connector(VenueType.KALSHI, reconnect_delay=0.1)
        
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.send = AsyncMock()
            
            messages = [
                # Created
                json.dumps({
                    "channel": "markets",
                    "type": "event",
                    "data": {
                        "event_ticker": "MARKET-NEW",
                        "title": "New Market",
                        "status": "open"
                    }
                }),
                # Updated
                json.dumps({
                    "channel": "markets",
                    "type": "update",
                    "data": {
                        "event_ticker": "MARKET-UPD",
                        "title": "Updated Market",
                        "status": "open"
                    }
                }),
                # Resolved
                json.dumps({
                    "channel": "markets",
                    "type": "event",
                    "data": {
                        "event_ticker": "MARKET-RES",
                        "title": "Resolved Market",
                        "status": "resolved"
                    }
                }),
            ]
            
            async def mock_iter():
                for msg in messages:
                    yield msg
                connector._running = False
            
            mock_ws.__aiter__ = lambda self: mock_iter()
            
            mock_connect.return_value = mock_ws
            
            await connector.connect()
            
            events = []
            async for event in connector.stream_events():
                if deduplicator.is_duplicate(event):
                    continue
                events.append(event)
            
            assert len(events) == 3
            assert events[0].event_type == EventType.CREATED
            assert events[1].event_type == EventType.UPDATED
            assert events[2].event_type == EventType.RESOLVED
            
            await connector.disconnect()

