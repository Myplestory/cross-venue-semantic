"""Tests for discovery types."""

import pytest
from datetime import datetime

from ..types import VenueType, EventType, MarketEvent, OutcomeSpec


def test_market_event_creation():
    """Test MarketEvent creation with defaults."""
    event = MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-123",
        event_type=EventType.CREATED,
        title="Test Market"
    )
    
    assert event.venue == VenueType.KALSHI
    assert event.venue_market_id == "MARKET-123"
    assert event.event_type == EventType.CREATED
    assert event.title == "Test Market"
    assert event.outcomes == []
    assert event.raw_payload == {}
    assert event.received_at is not None


def test_market_event_with_outcomes():
    """Test MarketEvent with outcomes."""
    outcomes = [
        OutcomeSpec(outcome_id="YES", label="Yes"),
        OutcomeSpec(outcome_id="NO", label="No")
    ]
    
    event = MarketEvent(
        venue=VenueType.POLYMARKET,
        venue_market_id="0x123",
        event_type=EventType.CREATED,
        title="Test Market",
        outcomes=outcomes
    )
    
    assert len(event.outcomes) == 2
    assert event.outcomes[0].outcome_id == "YES"
    assert event.outcomes[1].label == "No"


def test_market_event_with_end_date():
    """Test MarketEvent with end date."""
    end_date = datetime(2024, 12, 31, 23, 59, 59)
    
    event = MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-123",
        event_type=EventType.CREATED,
        title="Test Market",
        end_date=end_date
    )
    
    assert event.end_date == end_date

