"""Performance and stress tests for canonicalization module."""

import pytest
import asyncio
import time
from datetime import datetime
from discovery.types import MarketEvent, VenueType, EventType, OutcomeSpec
from canonicalization.text_builder import KalshiTextBuilder, get_builder
from canonicalization.hasher import ContentHasher
from canonicalization.types import CanonicalEvent


class TestTextBuilderPerformance:
    """Performance tests for text builders."""
    
    def test_build_single_event_performance(self):
        """Test single event build performance."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="PERF-1",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000?",
            description="Test description",
            resolution_criteria="Test criteria",
            end_date=datetime(2024, 12, 31),
        )
        
        builder = KalshiTextBuilder()
        
        start = time.perf_counter()
        text = builder.build(event)
        elapsed = time.perf_counter() - start
        
        # Should be very fast (<10ms)
        assert elapsed < 0.01
        assert len(text) > 0
    
    @pytest.mark.asyncio
    async def test_build_batch_performance(self):
        """Test batch build performance."""
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=f"PERF-{i}",
                event_type=EventType.CREATED,
                title=f"Market {i} question?",
            )
            for i in range(100)
        ]
        
        builder = KalshiTextBuilder()
        
        start = time.perf_counter()
        results = await builder.build_batch(events)
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (<1 second for 100 events)
        assert elapsed < 1.0
        assert len(results) == 100
    
    @pytest.mark.asyncio
    async def test_build_large_batch_performance(self):
        """Test large batch build performance (1000 events)."""
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=f"LARGE-{i}",
                event_type=EventType.CREATED,
                title=f"Market {i} question?",
                description=f"Description for market {i}",
            )
            for i in range(1000)
        ]
        
        builder = KalshiTextBuilder()
        
        start = time.perf_counter()
        results = await builder.build_batch(events)
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (<10 seconds for 1000 events)
        assert elapsed < 10.0
        assert len(results) == 1000


class TestHasherPerformance:
    """Performance tests for hasher."""
    
    def test_hash_single_text_performance(self):
        """Test single text hash performance."""
        text = "Market Statement:\nWill Bitcoin reach $100,000?"
        
        start = time.perf_counter()
        hash_val = ContentHasher.hash_content(text)
        elapsed = time.perf_counter() - start
        
        # Should be very fast (<1ms)
        assert elapsed < 0.001
        assert len(hash_val) == 64
    
    @pytest.mark.asyncio
    async def test_hash_batch_performance(self):
        """Test batch hash performance."""
        texts = [f"Market {i} statement" for i in range(100)]
        
        start = time.perf_counter()
        hashes = await ContentHasher.hash_batch(texts)
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (<1 second for 100 texts)
        assert elapsed < 1.0
        assert len(hashes) == 100
    
    @pytest.mark.asyncio
    async def test_hash_large_batch_performance(self):
        """Test large batch hash performance (1000 texts)."""
        texts = [f"Market {i} statement" for i in range(1000)]
        
        start = time.perf_counter()
        hashes = await ContentHasher.hash_batch(texts)
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (<5 seconds for 1000 texts)
        assert elapsed < 5.0
        assert len(hashes) == 1000
    
    def test_hash_very_long_text_performance(self):
        """Test hash performance with very long text."""
        long_text = "A" * 100000  # 100KB
        
        start = time.perf_counter()
        hash_val = ContentHasher.hash_content(long_text)
        elapsed = time.perf_counter() - start
        
        # Should still be fast (<10ms even for 100KB)
        assert elapsed < 0.01
        assert len(hash_val) == 64


class TestFullPipelinePerformance:
    """Performance tests for full canonicalization pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_canonicalization_performance(self):
        """Test full canonicalization flow performance."""
        event = MarketEvent(
            venue=VenueType.KALSHI,
            venue_market_id="FULL-1",
            event_type=EventType.CREATED,
            title="Will Bitcoin reach $100,000?",
            description="Test description",
            resolution_criteria="Test criteria",
            end_date=datetime(2024, 12, 31),
        )
        
        builder = get_builder(event.venue)
        
        start = time.perf_counter()
        
        # Build canonical text
        canonical_text = await builder.build_async(event)
        
        # Generate hashes
        content_hash = await ContentHasher.hash_content_async(canonical_text)
        identity_hash = ContentHasher.identity_hash(
            event.venue,
            event.venue_market_id
        )
        
        # Create canonical event
        canonical_event = CanonicalEvent(
            event=event,
            canonical_text=canonical_text,
            content_hash=content_hash,
            identity_hash=identity_hash,
        )
        
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (<50ms)
        assert elapsed < 0.05
        assert canonical_event is not None
        assert len(canonical_event.content_hash) == 64
    
    @pytest.mark.asyncio
    async def test_batch_full_canonicalization_performance(self):
        """Test batch full canonicalization performance."""
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=f"BATCH-FULL-{i}",
                event_type=EventType.CREATED,
                title=f"Market {i} question?",
            )
            for i in range(100)
        ]
        
        builder = get_builder(VenueType.KALSHI)
        
        start = time.perf_counter()
        
        # Build texts in parallel
        event_texts = await builder.build_batch(events)
        
        # Hash texts in parallel
        texts = [text for _, text in event_texts]
        hashes = await ContentHasher.hash_batch(texts)
        
        # Create canonical events
        canonical_events = [
            CanonicalEvent(
                event=event,
                canonical_text=text,
                content_hash=hash_val,
                identity_hash=ContentHasher.identity_hash(
                    event.venue,
                    event.venue_market_id
                ),
            )
            for (event, text), hash_val in zip(event_texts, hashes)
        ]
        
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (<2 seconds for 100 events)
        assert elapsed < 2.0
        assert len(canonical_events) == 100


class TestConcurrentOperations:
    """Tests for concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_builds(self):
        """Test concurrent text building."""
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=f"CONC-{i}",
                event_type=EventType.CREATED,
                title=f"Market {i} question?",
            )
            for i in range(50)
        ]
        
        builder = KalshiTextBuilder()
        
        # Create multiple concurrent tasks
        tasks = [builder.build_async(event) for event in events]
        
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        
        # Should complete efficiently with concurrency
        assert elapsed < 1.0
        assert len(results) == 50
        assert all(isinstance(text, str) for text in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_hashes(self):
        """Test concurrent hashing."""
        texts = [f"Market {i} statement" for i in range(50)]
        
        # Create multiple concurrent tasks
        tasks = [ContentHasher.hash_content_async(text) for text in texts]
        
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        
        # Should complete efficiently with concurrency
        assert elapsed < 1.0
        assert len(results) == 50
        assert all(isinstance(h, str) for h in results)
        assert all(len(h) == 64 for h in results)


class TestMemoryEfficiency:
    """Tests for memory efficiency."""
    
    @pytest.mark.asyncio
    async def test_batch_memory_efficiency(self):
        """Test that batch operations don't consume excessive memory."""
        # Create large batch
        events = [
            MarketEvent(
                venue=VenueType.KALSHI,
                venue_market_id=f"MEM-{i}",
                event_type=EventType.CREATED,
                title=f"Market {i} question?",
            )
            for i in range(1000)
        ]
        
        builder = KalshiTextBuilder()
        
        # Process in batches
        batch_size = 100
        all_results = []
        
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            results = await builder.build_batch(batch)
            all_results.extend(results)
        
        assert len(all_results) == 1000
        assert all(isinstance(text, str) for _, text in all_results)

