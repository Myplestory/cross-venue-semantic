"""
Venue connector factory.

Creates and manages venue connectors dynamically.
Supports easy addition of new venues.
"""

import logging
from typing import Dict, Optional, Type
from .base_connector import BaseVenueConnector
from .types import VenueType
from .kalshi_poller import KalshiConnector
from .polymarket_poller import PolymarketConnector

logger = logging.getLogger(__name__)


# Registry of venue connectors
_VENUE_REGISTRY: Dict[VenueType, Type[BaseVenueConnector]] = {
    VenueType.KALSHI: KalshiConnector,
    VenueType.POLYMARKET: PolymarketConnector,
}


def register_venue(venue_type: VenueType, connector_class: Type[BaseVenueConnector]):
    """
    Register a new venue connector.
    
    Args:
        venue_type: Venue identifier
        connector_class: Connector class that inherits from BaseVenueConnector
        
    Example:
        from discovery.venue_factory import register_venue
        from discovery.types import VenueType
        
        class OpinionConnector(BaseVenueConnector):
            ...
        
        register_venue(VenueType.OPINION, OpinionConnector)
    """
    _VENUE_REGISTRY[venue_type] = connector_class
    logger.info(f"Registered venue connector: {venue_type.value}")


def create_connector(
    venue_type: VenueType,
    ws_url: Optional[str] = None,
    **kwargs
) -> BaseVenueConnector:
    """
    Create a venue connector instance.
    
    Args:
        venue_type: Venue to connect to
        ws_url: Optional WebSocket URL (uses default if not provided)
        **kwargs: Additional connector-specific arguments
        
    Returns:
        Configured connector instance
        
    Raises:
        ValueError: If venue type is not registered
        
    Example:
        from discovery.venue_factory import create_connector
        from discovery.types import VenueType
        
        connector = create_connector(
            VenueType.POLYMARKET,
            reconnect_delay=5.0
        )
    """
    if venue_type not in _VENUE_REGISTRY:
        raise ValueError(
            f"Venue {venue_type.value} is not registered. "
            f"Available venues: {[v.value for v in _VENUE_REGISTRY.keys()]}"
        )
    
    connector_class = _VENUE_REGISTRY[venue_type]
    
    # Use default URL if not provided
    if ws_url is None:
        ws_url = _get_default_url(venue_type)
    
    return connector_class(ws_url=ws_url, **kwargs)


def _get_default_url(venue_type: VenueType) -> str:
    """Get default WebSocket URL for venue."""
    defaults = {
        VenueType.KALSHI: "wss://api.kalshi.com/trade-api/v2/websocket",
        VenueType.POLYMARKET: "wss://ws-subscriptions-clob.polymarket.com/ws/market",
    }
    return defaults.get(venue_type, "")


def list_available_venues() -> list[VenueType]:
    """List all registered venue types."""
    return list(_VENUE_REGISTRY.keys())

