"""Integration tests for canonicalization module."""

import pytest
import asyncio
from datetime import datetime
from discovery.types import MarketEvent, VenueType, EventType, OutcomeSpec
from canonicalization.text_builder import get_builder
from canonicalization.hasher import ContentHasher
from canonicalization.types import CanonicalEvent


@pytest.fixture
def sample_event():
    """Create sample market event."""
    return MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="MARKET-123",
        event_type=EventType.CREATED,
        title="Will Bitcoin reach $100,000 by Dec 31, 2024?",
        description="This market resolves based on Coinbase closing price",
        resolution_criteria="Resolves YES if Bitcoin closes above $100,000 on Dec 31, 2024",
        end_date=datetime(2024, 12, 31),
        outcomes=[
            OutcomeSpec(outcome_id="YES", label="Yes"),
            OutcomeSpec(outcome_id="NO", label="No"),
        ],
    )


class TestCanonicalizationFlow:
    """Tests for full canonicalization flow."""
    
    @pytest.mark.asyncio
    async def test_full_canonicalization_flow(self, sample_event):
        """Test complete flow: event → canonical text → hash."""
        # Get builder
        builder = get_builder(sample_event.venue)
        
        # Build canonical text
        canonical_text = await builder.build_async(sample_event)
        
        # Generate hashes
        content_hash = await ContentHasher.hash_content_async(canonical_text)
        identity_hash = ContentHasher.identity_hash(
            sample_event.venue,
            sample_event.venue_market_id
        )
        
        # Create canonical event
        canonical_event = CanonicalEvent(
            event=sample_event,
            canonical_text=canonical_text,
            content_hash=content_hash,
            identity_hash=identity_hash,
        )
        
        # Verify
        assert canonical_event.event == sample_event
        assert len(canonical_event.canonical_text) > 0
        assert len(canonical_event.content_hash) == 64
        assert len(canonical_event.identity_hash) == 64
        assert canonical_event.created_at is not None
    
    @pytest.mark.asyncio
    async def test_batch_canonicalization(self):
        """Test batch canonicalization (non-blocking)."""
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=f"MARKET-{i}",
                event_type=EventType.CREATED,
                title=f"Market {i} question?",
            )
            for i in range(5)
        ]
        
        builder = get_builder(VenueType.KALSHI)
        
        # Build texts in parallel
        event_texts = await builder.build_batch(events)
        
        # Hash texts in parallel
        texts = [text for _, text in event_texts]
        hashes = await ContentHasher.hash_batch(texts)
        
        # Verify
        assert len(event_texts) == 5
        assert len(hashes) == 5
        assert all(len(h) == 64 for h in hashes)
    
    @pytest.mark.asyncio
    async def test_change_detection(self, sample_event):
        """Test content hash detects changes."""
        builder = get_builder(sample_event.venue)
        
        # Original event
        text1 = await builder.build_async(sample_event)
        hash1 = await ContentHasher.hash_content_async(text1)
        
        # Updated event (description changed)
        sample_event.description = "Updated description"
        text2 = await builder.build_async(sample_event)
        hash2 = await ContentHasher.hash_content_async(text2)
        
        # Hashes should be different
        assert hash1 != hash2
    
    @pytest.mark.asyncio
    async def test_no_change_detection(self, sample_event):
        """Test content hash detects no change."""
        builder = get_builder(sample_event.venue)
        
        # Build text twice
        text1 = await builder.build_async(sample_event)
        hash1 = await ContentHasher.hash_content_async(text1)
        
        text2 = await builder.build_async(sample_event)
        hash2 = await ContentHasher.hash_content_async(text2)
        
        # Hashes should be same
        assert hash1 == hash2


class TestMultiVenueCanonicalization:
    """Tests for canonicalization across multiple venues."""
    
    @pytest.mark.asyncio
    async def test_kalshi_and_polymarket(self):
        """Test canonicalization for both venues."""
        kalshi_event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="KALSHI-123",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000?",
        )
        
        poly_event = MarketEvent(
            venue=VenueType.POLYMARKET,
            venue_market_id="0xpoly123",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000?",
        )
        
        kalshi_builder = get_builder(VenueType.KALSHI)
        poly_builder = get_builder(VenueType.POLYMARKET)
        
        kalshi_text = await kalshi_builder.build_async(kalshi_event)
        poly_text = await poly_builder.build_async(poly_event)
        
        # Both should produce canonical text
        assert len(kalshi_text) > 0
        assert len(poly_text) > 0
        
        # Both should have same structure
        assert "Market Statement:" in kalshi_text
        assert "Market Statement:" in poly_text

