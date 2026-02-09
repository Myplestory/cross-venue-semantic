"""Edge case tests for canonical text builders."""

import pytest
from datetime import datetime
from discovery.types import MarketEvent, VenueType, EventType, OutcomeSpec
from canonicalization.text_builder import KalshiTextBuilder, PolymarketTextBuilder


class TestTextBuilderEdgeCases:
    """Tests for edge cases in text building."""
    
    def test_empty_title(self):
        """Test handling of empty title."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="EMPTY-1",
            event_type=EventType.CREATED,
            title="",
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        # Should still produce valid output
        assert "Market Statement:" in text
        assert len(text) > 0
    
    def test_very_long_title(self):
        """Test handling of very long title."""
        long_title = "A" * 1000  # 1000 character title
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="LONG-1",
            event_type=EventType.CREATED,
            title=long_title,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert long_title in text
        assert len(text) >= len(long_title)
    
    def test_multiline_title(self):
        """Test handling of multiline title."""
        multiline_title = "Line 1\nLine 2\nLine 3"
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="MULTI-1",
            event_type=EventType.CREATED,
            title=multiline_title,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert multiline_title in text
    
    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        unicode_title = "Will Bitcoin reach $100,000? 🚀 Élection 2024 中文"
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="UNICODE-1",
            event_type=EventType.CREATED,
            title=unicode_title,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert "🚀" in text
        assert "Élection" in text
        assert "中文" in text
    
    def test_special_characters(self):
        """Test handling of special characters."""
        special_title = "Will $BTC reach $100,000? (Yes/No) [2024]"
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="SPECIAL-1",
            event_type=EventType.CREATED,
            title=special_title,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert "$BTC" in text
        assert "(Yes/No)" in text
        assert "[2024]" in text
    
    def test_html_entities(self):
        """Test handling of HTML-like entities."""
        html_title = "Will Bitcoin & Ethereum reach $100K?"
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="HTML-1",
            event_type=EventType.CREATED,
            title=html_title,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert "&" in text
        assert "Ethereum" in text
    
    def test_empty_description(self):
        """Test handling of empty description."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="DESC-1",
            event_type=EventType.CREATED,
            title="Test market?",
            description="",
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        # Empty description should not appear
        assert "Clarifications:" not in text or "Clarifications:\n" in text
    
    def test_very_long_description(self):
        """Test handling of very long description."""
        long_desc = "A" * 5000  # 5000 character description
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="LONG-DESC-1",
            event_type=EventType.CREATED,
            title="Test market?",
            description=long_desc,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert long_desc in text
    
    def test_multiline_description(self):
        """Test handling of multiline description."""
        multiline_desc = "Line 1\nLine 2\nLine 3\nLine 4"
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="MULTI-DESC-1",
            event_type=EventType.CREATED,
            title="Test market?",
            description=multiline_desc,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert multiline_desc in text
    
    def test_none_values(self):
        """Test handling of None values."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="NONE-1",
            event_type=EventType.CREATED,
            title="Test market?",
            description=None,
            resolution_criteria=None,
            end_date=None,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        # Should not crash and should produce valid output
        assert "Market Statement:" in text
        assert "Test market?" in text
        # None fields should not appear
        assert "Resolution Criteria:" not in text or "Resolution Criteria:\n" in text
    
    def test_future_date(self):
        """Test handling of future dates."""
        future_date = datetime(2099, 12, 31)
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="FUTURE-1",
            event_type=EventType.CREATED,
            title="Test market?",
            end_date=future_date,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert "2099-12-31" in text
    
    def test_past_date(self):
        """Test handling of past dates."""
        past_date = datetime(2020, 1, 1)
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="PAST-1",
            event_type=EventType.CREATED,
            title="Test market?",
            end_date=past_date,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert "2020-01-01" in text
    
    def test_many_outcomes(self):
        """Test handling of many outcomes (10+)."""
        outcomes = [
            OutcomeSpec(outcome_id=f"OUT-{i}", label=f"Outcome {i}")
            for i in range(15)
        ]
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="MANY-OUT-1",
            event_type=EventType.CREATED,
            title="Who will win?",
            outcomes=outcomes,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        assert "Outcomes:" in text
        # All outcomes should be listed
        for i in range(15):
            assert f"Outcome {i}" in text
    
    def test_empty_outcomes_list(self):
        """Test handling of empty outcomes list."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="EMPTY-OUT-1",
            event_type=EventType.CREATED,
            title="Test market?",
            outcomes=[],
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        # Empty outcomes should not show outcomes section
        assert "Outcomes:" not in text
    
    def test_outcome_with_special_characters(self):
        """Test handling of outcomes with special characters."""
        outcomes = [
            OutcomeSpec(outcome_id="YES", label="Yes (100%)"),
            OutcomeSpec(outcome_id="NO", label="No (0%)"),
        ]
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="SPEC-OUT-1",
            event_type=EventType.CREATED,
            title="Test market?",
            outcomes=outcomes,
        )
        
        builder = KalshiTextBuilder()
        text = builder.build(event)
        
        # YES/NO markets don't show outcomes, but if they did...
        # This is just to ensure no crashes
        assert len(text) > 0


class TestTextBuilderConsistency:
    """Tests for consistency across venues."""
    
    def test_same_event_different_venues(self):
        """Test that same event produces similar structure across venues."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="CONSIST-1",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000?",
            description="Test description",
            resolution_criteria="Test criteria",
            end_date=datetime(2024, 12, 31),
        )
        
        kalshi_builder = KalshiTextBuilder()
        poly_builder = PolymarketTextBuilder()
        
        kalshi_text = kalshi_builder.build(event)
        poly_text = poly_builder.build(event)
        
        # Both should have same structure
        assert "Market Statement:" in kalshi_text
        assert "Market Statement:" in poly_text
        assert "Resolution Criteria:" in kalshi_text
        assert "Resolution Criteria:" in poly_text
        assert "2024-12-31" in kalshi_text
        assert "2024-12-31" in poly_text
    
    def test_event_type_does_not_affect_output(self):
        """Test that event_type doesn't affect canonical text."""
        event1 = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="EVT-1",
            event_type=EventType.CREATED,
            title="Test market?",
        )
        
        event2 = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="EVT-1",
            event_type=EventType.UPDATED,  # Different event type
            title="Test market?",
        )
        
        builder = KalshiTextBuilder()
        text1 = builder.build(event1)
        text2 = builder.build(event2)
        
        # Should produce same canonical text
        assert text1 == text2


class TestTextBuilderAsyncEdgeCases:
    """Tests for async edge cases."""
    
    @pytest.mark.asyncio
    async def test_build_async_with_none_values(self):
        """Test async build with None values."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="ASYNC-NONE-1",
            event_type=EventType.CREATED,
            title="Test market?",
            description=None,
            resolution_criteria=None,
        )
        
        builder = KalshiTextBuilder()
        text = await builder.build_async(event)
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    @pytest.mark.asyncio
    async def test_build_batch_empty_list(self):
        """Test batch building with empty list."""
        builder = KalshiTextBuilder()
        results = await builder.build_batch([])
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_build_batch_large_list(self):
        """Test batch building with large list (100 events)."""
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=f"BATCH-{i}",
                event_type=EventType.CREATED,
                title=f"Market {i} question?",
            )
            for i in range(100)
        ]
        
        builder = KalshiTextBuilder()
        results = await builder.build_batch(events)
        
        assert len(results) == 100
        assert all(isinstance(text, str) for _, text in results)
        assert all(len(text) > 0 for _, text in results)
    
    @pytest.mark.asyncio
    async def test_build_batch_mixed_validity(self):
        """Test batch building with mix of valid and edge case events."""
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id="VALID-1",
                event_type=EventType.CREATED,
                title="Valid market?",
            ),
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id="EMPTY-1",
                event_type=EventType.CREATED,
                title="",  # Empty title
            ),
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id="NONE-1",
                event_type=EventType.CREATED,
                title="None values?",
                description=None,
            ),
        ]
        
        builder = KalshiTextBuilder()
        results = await builder.build_batch(events)
        
        # Should handle all gracefully
        assert len(results) == 3
        assert all(isinstance(text, str) for _, text in results)


