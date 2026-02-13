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


def _kalshi_kwargs():
    """Kalshi auth from config (optional)."""
    try:
        import config
        kwargs = {}
        key_id = getattr(config, "KALSHI_API_KEY_ID", None)
        if key_id and str(key_id).strip():
            kwargs["api_key_id"] = str(key_id).strip()
        path = getattr(config, "KALSHI_PRIVATE_KEY_PATH", None)
        if path and str(path).strip():
            # Normalize: forward slashes work on Windows; avoid backslash escape issues from .env
            kwargs["private_key_path"] = str(path).strip().replace("\\", "/")
        pem = getattr(config, "KALSHI_PRIVATE_KEY", None)
        if pem and str(pem).strip():
            kwargs["private_key_pem"] = str(pem).strip()
        return kwargs
    except ImportError:
        return {}


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
    
    # Kalshi: inject auth from config if not passed
    if venue_type == VenueType.KALSHI:
        kalshi_cfg = _kalshi_kwargs()
        for k, v in kalshi_cfg.items():
            if k not in kwargs:
                kwargs[k] = v

    # Polymarket: inject Gamma API URL from config if not passed
    if venue_type == VenueType.POLYMARKET and "gamma_api_url" not in kwargs:
        try:
            import config as _cfg
            gamma_url = getattr(_cfg, "POLYMARKET_GAMMA_API_URL", None)
            if gamma_url and str(gamma_url).strip():
                kwargs["gamma_api_url"] = str(gamma_url).strip()
        except ImportError:
            pass

    return connector_class(ws_url=ws_url, **kwargs)


def _get_default_url(venue_type: VenueType) -> str:
    """Get default WebSocket URL for venue."""
    if venue_type == VenueType.KALSHI:
        try:
            import config
            url = getattr(config, "KALSHI_WS_URL", None)
            if url and str(url).strip():
                return str(url).strip()
            if getattr(config, "KALSHI_USE_DEMO", False):
                return "wss://demo-api.kalshi.co/trade-api/ws/v2"
        except ImportError:
            pass
        return "wss://api.elections.kalshi.com/trade-api/ws/v2"
    defaults = {
        VenueType.POLYMARKET: "wss://ws-subscriptions-clob.polymarket.com/ws/market",
    }
    return defaults.get(venue_type, "")


def list_available_venues() -> list[VenueType]:
    """List all registered venue types."""
    return list(_VENUE_REGISTRY.keys())

