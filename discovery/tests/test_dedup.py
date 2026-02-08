"""Tests for deduplication module."""

import pytest
from datetime import datetime, timedelta

from ..dedup import MarketDeduplicator
from ..types import MarketEvent, VenueType, EventType


def test_deduplicator_identity_hash():
    """Test identity hash generation."""
    dedup = MarketDeduplicator()
    
    event1 = MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-123",
        event_type=EventType.CREATED,
        title="Test Market"
    )
    
    event2 = MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-123",
        event_type=EventType.UPDATED,  # Different event type
        title="Test Market Updated"  # Different title
    )
    
    # Same venue + market_id = same hash = duplicate
    assert dedup.is_duplicate(event1) is False  # First time
    assert dedup.is_duplicate(event2) is True  # Duplicate (same identity)


def test_deduplicator_different_markets():
    """Test that different markets are not duplicates."""
    dedup = MarketDeduplicator()
    
    event1 = MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-123",
        event_type=EventType.CREATED,
        title="Market 1"
    )
    
    event2 = MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-456",  # Different market
        event_type=EventType.CREATED,
        title="Market 2"
    )
    
    assert dedup.is_duplicate(event1) is False
    assert dedup.is_duplicate(event2) is False  # Not duplicate


def test_deduplicator_different_venues():
    """Test that same market ID in different venues are not duplicates."""
    dedup = MarketDeduplicator()
    
    event1 = MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-123",
        event_type=EventType.CREATED,
        title="Market"
    )
    
    event2 = MarketEvent(
        venue=VenueType.POLYMARKET,  # Different venue
        venue_market_id="MARKET-123",  # Same ID
        event_type=EventType.CREATED,
        title="Market"
    )
    
    assert dedup.is_duplicate(event1) is False
    assert dedup.is_duplicate(event2) is False  # Not duplicate (different venue)


def test_deduplicator_ttl_expiration():
    """Test TTL expiration cleanup."""
    dedup = MarketDeduplicator(ttl_seconds=1)  # 1 second TTL
    
    event = MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-123",
        event_type=EventType.CREATED,
        title="Test Market"
    )
    
    assert dedup.is_duplicate(event) is False
    assert dedup.is_duplicate(event) is True  # Still in cache
    
    # Wait for expiration
    import time
    time.sleep(1.1)
    
    # Manually trigger cleanup (in real usage, this happens automatically)
    dedup._cleanup_expired()
    
    # After expiration, should not be duplicate
    assert dedup.is_duplicate(event) is False


def test_deduplicator_clear():
    """Test clearing deduplicator."""
    dedup = MarketDeduplicator()
    
    event = MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-123",
        event_type=EventType.CREATED,
        title="Test Market"
    )
    
    assert dedup.is_duplicate(event) is False
    assert dedup.size() == 1
    
    dedup.clear()
    assert dedup.size() == 0
    assert dedup.is_duplicate(event) is False  # Not duplicate after clear

