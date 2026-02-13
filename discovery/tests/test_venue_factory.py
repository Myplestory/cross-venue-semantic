"""Tests for venue factory."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.venue_factory import (
    create_connector,
    register_venue,
    list_available_venues,
    _get_default_url
)
from discovery.types import VenueType
from discovery.kalshi_poller import KalshiConnector
from discovery.polymarket_poller import PolymarketConnector
from discovery.base_connector import BaseVenueConnector


def test_list_available_venues():
    """Test listing available venues."""
    venues = list_available_venues()
    
    assert VenueType.KALSHI in venues
    assert VenueType.POLYMARKET in venues
    assert len(venues) >= 2


def test_create_connector_kalshi():
    """Test creating Kalshi connector."""
    connector = create_connector(VenueType.KALSHI)
    
    assert isinstance(connector, KalshiConnector)
    assert connector.venue_name == VenueType.KALSHI
    assert connector.ws_url == "wss://api.elections.kalshi.com/trade-api/ws/v2"


def test_create_connector_polymarket():
    """Test creating Polymarket connector."""
    connector = create_connector(VenueType.POLYMARKET)
    
    assert isinstance(connector, PolymarketConnector)
    assert connector.venue_name == VenueType.POLYMARKET
    assert connector.ws_url == "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def test_create_connector_custom_url():
    """Test creating connector with custom URL."""
    custom_url = "wss://custom.example.com/ws"
    connector = create_connector(VenueType.KALSHI, ws_url=custom_url)
    
    assert connector.ws_url == custom_url


def test_create_connector_with_kwargs():
    """Test creating connector with additional kwargs."""
    connector = create_connector(
        VenueType.KALSHI,
        reconnect_delay=10.0,
        max_reconnect_attempts=5
    )
    
    assert connector.reconnect_delay == 10.0
    assert connector.max_reconnect_attempts == 5


def test_create_connector_invalid_venue():
    """Test creating connector for unregistered venue."""
    # Try to create connector for a venue that doesn't exist
    # We'll use a string that's not a valid VenueType
    with pytest.raises((ValueError, AttributeError)):
        # This will fail because "INVALID" is not a valid VenueType enum value
        create_connector(VenueType.OPINION)  # OPINION exists but may not be registered


def test_register_venue():
    """Test registering a new venue."""
    class TestConnector(BaseVenueConnector):
        def __init__(self, ws_url: str = "wss://test.com/ws", **kwargs):
            super().__init__(
                venue_name=VenueType.OPINION,
                ws_url=ws_url,
                **kwargs
            )
        
        def _build_subscription_message(self):
            return {}
        
        async def _parse_message(self, message: str):
            return None
    
    # Register new venue
    register_venue(VenueType.OPINION, TestConnector)
    
    # Verify it's registered
    venues = list_available_venues()
    assert VenueType.OPINION in venues
    
    # Can create connector
    connector = create_connector(VenueType.OPINION)
    assert isinstance(connector, TestConnector)


def test_get_default_url():
    """Test getting default URLs."""
    kalshi_url = _get_default_url(VenueType.KALSHI)
    assert kalshi_url == "wss://api.elections.kalshi.com/trade-api/ws/v2"
    
    polymarket_url = _get_default_url(VenueType.POLYMARKET)
    assert polymarket_url == "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def test_factory_preserves_connector_type():
    """Test that factory creates correct connector types."""
    kalshi = create_connector(VenueType.KALSHI)
    assert type(kalshi).__name__ == "KalshiConnector"
    
    polymarket = create_connector(VenueType.POLYMARKET)
    assert type(polymarket).__name__ == "PolymarketConnector"

