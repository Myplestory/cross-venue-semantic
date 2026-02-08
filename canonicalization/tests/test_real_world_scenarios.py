"""Real-world scenario tests for canonicalization module."""

import pytest
from datetime import datetime
from discovery.types import MarketEvent, VenueType, EventType, OutcomeSpec
from canonicalization.text_builder import get_builder
from canonicalization.hasher import ContentHasher
from canonicalization.types import CanonicalEvent
from canonicalization.contract_spec import ContractSpec, EntitySpec, ThresholdSpec, DateSpec


class TestRealWorldMarketScenarios:
    """Tests based on real-world market scenarios."""
    
    @pytest.mark.asyncio
    async def test_bitcoin_price_market(self):
        """Test canonicalization of Bitcoin price market."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="BTC-PRICE-2024",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000 by December 31, 2024?",
            description="This market resolves based on Coinbase closing price on Dec 31, 2024",
            resolution_criteria="Resolves YES if Bitcoin (BTC) closes at or above $100,000 on Coinbase on December 31, 2024",
            end_date=datetime(2024, 12, 31),
            outcomes=[
                OutcomeSpec(outcome_id="YES", label="Yes"),
                OutcomeSpec(outcome_id="NO", label="No"),
            ],
        )
        
        builder = get_builder(VenueType.KALSHI)
        canonical_text = await builder.build_async(event)
        content_hash = await ContentHasher.hash_content_async(canonical_text)
        
        # Verify structure
        assert "Market Statement:" in canonical_text
        assert "$100,000" in canonical_text
        assert "Resolution Criteria:" in canonical_text
        assert "Coinbase" in canonical_text
        assert "2024-12-31" in canonical_text
        assert len(content_hash) == 64
    
    @pytest.mark.asyncio
    async def test_election_market(self):
        """Test canonicalization of election market."""
        event = MarketEvent(
            venue=VenueType.POLYMARKET,
            venue_market_id="0xelection2024",
            event_type=EventType.CREATED,
            title="Will Donald Trump win the 2024 US Presidential Election?",
            description="Based on official election results from all 50 states",
            resolution_criteria="Resolves YES if Donald Trump wins a majority of Electoral College votes",
            end_date=datetime(2024, 11, 5),
            outcomes=[
                OutcomeSpec(outcome_id="0xyes", label="Yes"),
                OutcomeSpec(outcome_id="0xno", label="No"),
            ],
        )
        
        builder = get_builder(VenueType.KALSHI)
        canonical_text = await builder.build_async(event)
        content_hash = await ContentHasher.hash_content_async(canonical_text)
        
        # Verify structure
        assert "Market Statement:" in canonical_text
        assert "Donald Trump" in canonical_text
        assert "2024" in canonical_text
        assert "Electoral College" in canonical_text
        assert len(content_hash) == 64
    
    @pytest.mark.asyncio
    async def test_multi_outcome_market(self):
        """Test canonicalization of multi-outcome market."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="MULTI-2024",
            event_type=EventType.CREATED,
            title="Who will win the 2024 US Presidential Election?",
            description="Based on official election results",
            resolution_criteria="Resolves based on candidate with most Electoral College votes",
            end_date=datetime(2024, 11, 5),
            outcomes=[
                OutcomeSpec(outcome_id="TRUMP", label="Donald Trump"),
                OutcomeSpec(outcome_id="BIDEN", label="Joe Biden"),
                OutcomeSpec(outcome_id="OTHER", label="Other"),
            ],
        )
        
        builder = get_builder(VenueType.KALSHI)
        canonical_text = await builder.build_async(event)
        
        # Verify structure
        assert "Market Statement:" in canonical_text
        assert "Outcomes:" in canonical_text
        assert "Donald Trump" in canonical_text
        assert "Joe Biden" in canonical_text
        assert "Other" in canonical_text
    
    @pytest.mark.asyncio
    async def test_sports_market(self):
        """Test canonicalization of sports market."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="SPORTS-2024",
            event_type=EventType.CREATED,
            title="Will the Kansas City Chiefs win Super Bowl LVIII?",
            description="Based on official NFL game results",
            resolution_criteria="Resolves YES if Kansas City Chiefs win Super Bowl LVIII",
            end_date=datetime(2024, 2, 11),
            outcomes=[
                OutcomeSpec(outcome_id="YES", label="Yes"),
                OutcomeSpec(outcome_id="NO", label="No"),
            ],
        )
        
        builder = get_builder(VenueType.KALSHI)
        canonical_text = await builder.build_async(event)
        
        # Verify structure
        assert "Kansas City Chiefs" in canonical_text
        assert "Super Bowl LVIII" in canonical_text
        assert "2024-02-11" in canonical_text
    
    @pytest.mark.asyncio
    async def test_weather_market(self):
        """Test canonicalization of weather market."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="WEATHER-2024",
            event_type=EventType.CREATED,
            title="Will New York City receive more than 50 inches of snow in 2024?",
            description="Based on NOAA weather station data",
            resolution_criteria="Resolves YES if total snowfall in NYC exceeds 50 inches in 2024",
            end_date=datetime(2024, 12, 31),
            outcomes=[
                OutcomeSpec(outcome_id="YES", label="Yes"),
                OutcomeSpec(outcome_id="NO", label="No"),
            ],
        )
        
        builder = get_builder(VenueType.KALSHI)
        canonical_text = await builder.build_async(event)
        
        # Verify structure
        assert "New York City" in canonical_text
        assert "50 inches" in canonical_text
        assert "NOAA" in canonical_text


class TestMarketUpdateScenarios:
    """Tests for market update scenarios."""
    
    @pytest.mark.asyncio
    async def test_market_description_update(self):
        """Test that description updates are detected via content hash."""
        event1 = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="UPDATE-1",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000?",
            description="Original description",
        )
        
        event2 = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="UPDATE-1",
            event_type=EventType.UPDATED,
            title="Will Bitcoin reach $100,000?",
            description="Updated description",  # Changed
        )
        
        builder = get_builder(VenueType.KALSHI)
        
        text1 = await builder.build_async(event1)
        text2 = await builder.build_async(event2)
        
        hash1 = await ContentHasher.hash_content_async(text1)
        hash2 = await ContentHasher.hash_content_async(text2)
        
        # Hashes should be different (content changed)
        assert hash1 != hash2
    
    @pytest.mark.asyncio
    async def test_market_title_update(self):
        """Test that title updates are detected via content hash."""
        event1 = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="UPDATE-2",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000?",
        )
        
        event2 = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="UPDATE-2",
            event_type=EventType.UPDATED,
            title="Will Bitcoin reach $200,000?",  # Changed
        )
        
        builder = get_builder(VenueType.KALSHI)
        
        text1 = await builder.build_async(event1)
        text2 = await builder.build_async(event2)
        
        hash1 = await ContentHasher.hash_content_async(text1)
        hash2 = await ContentHasher.hash_content_async(text2)
        
        # Hashes should be different (content changed)
        assert hash1 != hash2
    
    @pytest.mark.asyncio
    async def test_market_no_change_detection(self):
        """Test that unchanged markets produce same hash."""
        event1 = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="NOCHANGE-1",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000?",
            description="Test description",
        )
        
        event2 = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="NOCHANGE-1",
            event_type=EventType.UPDATED,  # Different event type
            title="Will Bitcoin reach $100,000?",  # Same content
            description="Test description",  # Same content
        )
        
        builder = get_builder(VenueType.KALSHI)
        
        text1 = await builder.build_async(event1)
        text2 = await builder.build_async(event2)
        
        hash1 = await ContentHasher.hash_content_async(text1)
        hash2 = await ContentHasher.hash_content_async(text2)
        
        # Hashes should be same (content unchanged)
        assert hash1 == hash2


class TestCrossVenueMatchingScenarios:
    """Tests for cross-venue matching scenarios."""
    
    @pytest.mark.asyncio
    async def test_same_market_different_venues(self):
        """Test canonicalization of same market from different venues."""
        kalshi_event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="KALSHI-BTC",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000 by Dec 31, 2024?",
            description="Coinbase closing price",
            resolution_criteria="Resolves YES if BTC closes above $100,000",
            end_date=datetime(2024, 12, 31),
        )
        
        poly_event = MarketEvent(
            venue=VenueType.POLYMARKET,
            venue_market_id="0xpolybtc",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000 by Dec 31, 2024?",
            description="Coinbase closing price",
            resolution_criteria="Resolves YES if BTC closes above $100,000",
            end_date=datetime(2024, 12, 31),
        )
        
        kalshi_builder = get_builder(VenueType.KALSHI)
        poly_builder = get_builder(VenueType.POLYMARKET)
        
        kalshi_text = await kalshi_builder.build_async(kalshi_event)
        poly_text = await poly_builder.build_async(poly_event)
        
        kalshi_hash = await ContentHasher.hash_content_async(kalshi_text)
        poly_hash = await ContentHasher.hash_content_async(poly_text)
        
        # Same content should produce same hash (for matching)
        assert kalshi_hash == poly_hash
        
        # But identity hashes should be different
        kalshi_identity = ContentHasher.identity_hash(
            VenueType.KALSHI,
            "KALSHI-BTC"
        )
        poly_identity = ContentHasher.identity_hash(
            VenueType.POLYMARKET,
            "0xpolybtc"
        )
        
        assert kalshi_identity != poly_identity


class TestContractSpecExtractionScenarios:
    """Tests for ContractSpec extraction scenarios."""
    
    def test_bitcoin_price_contract_spec(self):
        """Test ContractSpec for Bitcoin price market."""
        spec = ContractSpec(
            statement="Will Bitcoin reach $100,000 by December 31, 2024?",
            resolution_date=DateSpec(
                date=datetime(2024, 12, 31),
                is_deadline=True
            ),
            entities=[
                EntitySpec(name="Bitcoin", entity_type="other", aliases=["BTC"])
            ],
            thresholds=[
                ThresholdSpec(value=100000.0, unit="dollars", comparison=">=")
            ],
            resolution_criteria="Resolves YES if Bitcoin closes at or above $100,000 on Coinbase",
            data_source="Coinbase",
            outcome_labels=["Yes", "No"],
            confidence=0.95,
        )
        
        assert spec.statement == "Will Bitcoin reach $100,000 by December 31, 2024?"
        assert spec.resolution_date.date == datetime(2024, 12, 31)
        assert len(spec.entities) == 1
        assert spec.entities[0].name == "Bitcoin"
        assert len(spec.thresholds) == 1
        assert spec.thresholds[0].value == 100000.0
        assert spec.data_source == "Coinbase"
    
    def test_election_contract_spec(self):
        """Test ContractSpec for election market."""
        spec = ContractSpec(
            statement="Will Donald Trump win the 2024 US Presidential Election?",
            resolution_date=DateSpec(
                date=datetime(2024, 11, 5),
                is_deadline=True
            ),
            entities=[
                EntitySpec(name="Donald Trump", entity_type="person"),
                EntitySpec(name="United States", entity_type="location"),
            ],
            resolution_criteria="Resolves YES if Donald Trump wins majority of Electoral College votes",
            data_source="Official election results",
            outcome_labels=["Yes", "No"],
            confidence=0.98,
        )
        
        assert "Donald Trump" in spec.statement
        assert len(spec.entities) == 2
        assert spec.entities[0].entity_type == "person"
        assert spec.entities[1].entity_type == "location"
        assert spec.confidence == 0.98


class TestErrorRecoveryScenarios:
    """Tests for error recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_malformed_event_recovery(self):
        """Test recovery from malformed event."""
        # Event with None values
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="MALFORMED-1",
            event_type=EventType.CREATED,
            title="Test market?",
            description=None,
            resolution_criteria=None,
            end_date=None,
            outcomes=None,
        )
        
        builder = get_builder(VenueType.KALSHI)
        text = await builder.build_async(event)
        
        # Should handle gracefully
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Market Statement:" in text
    
    @pytest.mark.asyncio
    async def test_batch_with_errors(self):
        """Test batch processing with some errors."""
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=f"BATCH-ERR-{i}",
                event_type=EventType.CREATED,
                title=f"Market {i}?" if i != 5 else "",  # Empty title for one
            )
            for i in range(10)
        ]
        
        builder = get_builder(VenueType.KALSHI)
        results = await builder.build_batch(events)
        
        # Should handle gracefully, may skip invalid ones
        assert len(results) >= 9  # At least 9 valid results
        assert all(isinstance(text, str) for _, text in results)

