"""
Market Discovery Module

WebSocket-based market discovery from multiple venues.
Handles venue-specific connectors and event normalization.
"""

from .types import VenueType, EventType, MarketEvent, OutcomeSpec
from .venue_factory import create_connector, register_venue, list_available_venues
from .dedup import MarketDeduplicator
from .base_connector import BaseVenueConnector

__all__ = [
    "VenueType",
    "EventType",
    "MarketEvent",
    "OutcomeSpec",
    "create_connector",
    "register_venue",
    "list_available_venues",
    "MarketDeduplicator",
    "BaseVenueConnector",
]
