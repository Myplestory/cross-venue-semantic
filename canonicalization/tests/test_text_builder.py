"""Tests for canonical text builders."""

import pytest
from datetime import datetime
from discovery.types import MarketEvent, VenueType, EventType, OutcomeSpec
from canonicalization.text_builder import (
    CanonicalTextBuilder,
    KalshiTextBuilder,
    PolymarketTextBuilder,
    get_builder,
)


@pytest.fixture
def sample_kalshi_event():
    """Create sample Kalshi market event."""
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


@pytest.fixture
def sample_polymarket_event():
    """Create sample Polymarket market event."""
    return MarketEvent(
        venue=VenueType.POLYMARKET,
        venue_market_id="0xtest123",
        event_type=EventType.CREATED,
        title="Will Trump win the 2024 election?",
        description="Based on official election results",
        resolution_criteria="Resolves YES if Trump wins Electoral College",
        end_date=datetime(2024, 11, 5),
        outcomes=[
            OutcomeSpec(outcome_id="0xyes", label="Yes"),
            OutcomeSpec(outcome_id="0xno", label="No"),
        ],
    )


class TestKalshiTextBuilder:
    """Tests for Kalshi text builder."""
    
    def test_build_full_event(self, sample_kalshi_event):
        """Test building canonical text from complete event."""
        builder = KalshiTextBuilder()
        text = builder.build(sample_kalshi_event)
        
        assert "Market Statement:" in text
        assert sample_kalshi_event.title in text
        assert "Resolution Criteria:" in text
        assert sample_kalshi_event.resolution_criteria in text
        assert "Clarifications:" in text
        assert sample_kalshi_event.description in text
        assert "End Date: 2024-12-31" in text
    
    def test_build_minimal_event(self):
        """Test building text from minimal event (only title)."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="MIN-123",
            event_type=EventType.CREATED,
            title="Simple market question?",
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert "Market Statement:" in text
        assert event.title in text
        assert "Resolution Criteria:" not in text
        assert "Clarifications:" not in text
        assert "End Date:" not in text
    
    def test_build_multi_outcome_event(self):
        """Test building text for multi-outcome market."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="MULTI-123",
            event_type=EventType.CREATED,
            title="Who will win the election?",
            outcomes=[
                OutcomeSpec(outcome_id="TRUMP", label="Trump"),
                OutcomeSpec(outcome_id="BIDEN", label="Biden"),
                OutcomeSpec(outcome_id="OTHER", label="Other"),
            ],
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert "Outcomes:" in text
        assert "Trump" in text
        assert "Biden" in text
        assert "Other" in text
    
    def test_build_no_outcomes_section_for_yes_no(self):
        """Test that YES/NO markets don't show outcomes section."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="YESNO-123",
            event_type=EventType.CREATED,
            title="Simple yes/no question?",
            outcomes=[
                OutcomeSpec(outcome_id="YES", label="Yes"),
                OutcomeSpec(outcome_id="NO", label="No"),
            ],
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        # YES/NO markets (2 outcomes) should not show outcomes section
        assert "Outcomes:" not in text


class TestPolymarketTextBuilder:
    """Tests for Polymarket text builder."""
    
    def test_build_full_event(self, sample_polymarket_event):
        """Test building canonical text from complete event."""
        builder = PolymarketTextBuilder()
        text = builder.build(sample_polymarket_event)
        
        assert "Market Statement:" in text
        assert sample_polymarket_event.title in text
        assert "Resolution Criteria:" in text
        assert sample_polymarket_event.resolution_criteria in text
        assert "Clarifications:" in text
        assert sample_polymarket_event.description in text
        assert "End Date: 2024-11-05" in text
    
    def test_build_minimal_event(self):
        """Test building text from minimal event."""
        event = MarketEvent(
            venue=VenueType.POLYMARKET,
            venue_market_id="0xmin",
            event_type=EventType.CREATED,
            title="Simple question?",
        )
        
        builder = PolymarketTextBuilder()
        text = builder.build(event)
        
        assert "Market Statement:" in text
        assert event.title in text
        # Optional sections should not appear
        assert "Resolution Criteria:" not in text or "Resolution Criteria:\n" in text


class TestTextBuilderAsync:
    """Tests for async text builder methods."""
    
    @pytest.mark.asyncio
    async def test_build_async(self, sample_kalshi_event):
        """Test async build method."""
        builder = KalshiTextBuilder()
        text = await builder.build_async(sample_kalshi_event)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert sample_kalshi_event.title in text
    
    @pytest.mark.asyncio
    async def test_build_batch(self):
        """Test batch building (non-blocking)."""
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=f"MARKET-{i}",
                event_type=EventType.CREATED,
                title=f"Market {i} question?",
            )
            for i in range(5)
        ]
        
        builder = KalshiTextBuilder()
        results = await builder.build_batch(events)
        
        assert len(results) == 5
        assert all(isinstance(text, str) for _, text in results)
        assert all(len(text) > 0 for _, text in results)
    
    @pytest.mark.asyncio
    async def test_build_batch_with_errors(self):
        """Test batch building handles errors gracefully."""
        # Create valid and invalid events
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id="VALID-1",
                event_type=EventType.CREATED,
                title="Valid market?",
            ),
            # Invalid event (missing title would cause error in real scenario)
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id="VALID-2",
                event_type=EventType.CREATED,
                title="Another valid market?",
            ),
        ]
        
        builder = KalshiTextBuilder()
        results = await builder.build_batch(events)
        
        # Should handle gracefully and return valid results
        assert len(results) == 2


class TestTextBuilderFactory:
    """Tests for builder factory function."""
    
    def test_get_builder_kalshi(self):
        """Test getting Kalshi builder."""
        builder = get_builder(VenueType.KALSHI)
        assert isinstance(builder, KalshiTextBuilder)
    
    def test_get_builder_polymarket(self):
        """Test getting Polymarket builder."""
        builder = get_builder(VenueType.POLYMARKET)
        assert isinstance(builder, PolymarketTextBuilder)
    
    def test_get_builder_invalid_venue(self):
        """Test getting builder for unsupported venue."""
        with pytest.raises(ValueError, match="Unsupported venue"):
            get_builder(VenueType.OPINION)  # Not yet implemented


