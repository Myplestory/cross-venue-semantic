"""
Type definitions for market discovery.

Core event types and connector interfaces.
"""

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Literal, Optional, Protocol, AsyncIterator
from enum import Enum


class VenueType(str, Enum):
    """Supported trading venues."""
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"
    OPINION = "opinion"
    GEMINI = "gemini"


class EventType(str, Enum):
    """Market event types."""
    CREATED = "created"
    UPDATED = "updated"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class OutcomeSpec:
    """Outcome specification for a market."""
    outcome_id: str
    label: str  # "Yes", "No", "Trump", "Biden", etc.


@dataclass
class MarketEvent:
    """Normalized market event from any venue."""
    venue: VenueType
    venue_market_id: str  # Venue's native market ID
    event_type: EventType
    title: str
    description: Optional[str] = None
    resolution_criteria: Optional[str] = None
    end_date: Optional[datetime] = None
    outcomes: list[OutcomeSpec] = None  # YES/NO or multi-outcome
    raw_payload: dict = None  # Preserve original for debugging
    received_at: datetime = None

    def __post_init__(self):
        """Initialize defaults after dataclass creation."""
        if self.outcomes is None:
            self.outcomes = []
        if self.raw_payload is None:
            self.raw_payload = {}
        if self.received_at is None:
            self.received_at = datetime.now(UTC)


class VenueConnector(Protocol):
    """Protocol for venue connectors."""
    
    venue_name: VenueType
    
    async def connect(self) -> None:
        """Establish WebSocket connection to venue."""
        ...
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        ...
    
    async def stream_events(self) -> AsyncIterator[MarketEvent]:
        """Stream market events as they arrive via WebSocket."""
        ...

