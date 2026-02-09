"""Tests for EmbeddingProcessor orchestration."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from embedding.processor import EmbeddingProcessor
from embedding.encoder import EmbeddingEncoder
from embedding.index import QdrantIndex
from embedding.cache.in_memory import InMemoryCache


@pytest.mark.asyncio
async def test_process_with_cache_hit(sample_canonical_event, mock_embedding):
    """Test processing with cache hit."""
    # Mock encoder (should not be called on cache hit)
    encoder = EmbeddingEncoder()
    encoder.encode_async = AsyncMock()
    encoder.model_name = "Qwen/Qwen3-Embedding-4B"
    encoder.embedding_dim = 2048
    encoder._initialized = True  # Skip initialization
    
    # Mock index to avoid real Qdrant connection
    index = QdrantIndex()
    index.upsert_async = AsyncMock()  # Mock upsert to avoid real writes
    index.initialize = AsyncMock()  # Mock initialize to prevent real connection
    index._initialized = True  # Mark as initialized
    
    # Cache with pre-populated embedding
    cache = InMemoryCache()
    await cache.initialize()
    await cache.set(sample_canonical_event.content_hash, mock_embedding)
    
    processor = EmbeddingProcessor(encoder, index, cache)
    await processor.initialize()
    
    # Process event
    embedded_event = await processor.process_async(sample_canonical_event)
    
    # Verify encoder was NOT called (cache hit)
    encoder.encode_async.assert_not_called()
    
    # Verify Qdrant upsert was called
    index.upsert_async.assert_called_once()
    
    # Verify embedded event
    assert embedded_event.embedding == mock_embedding
    assert embedded_event.canonical_event == sample_canonical_event


@pytest.mark.asyncio
async def test_process_with_cache_miss(sample_canonical_event, mock_embedding):
    """Test processing with cache miss."""
    # Mock encoder (should be called on cache miss)
    encoder = EmbeddingEncoder()
    encoder.encode_async = AsyncMock(return_value=mock_embedding)
    encoder.model_name = "Qwen/Qwen3-Embedding-4B"
    encoder.embedding_dim = 2048
    encoder._initialized = True  # Skip initialization
    
    # Mock index
    index = QdrantIndex()
    index.upsert_async = AsyncMock()
    index.initialize = AsyncMock()  # Mock initialize to prevent real connection
    index._initialized = True  # Mark as initialized
    
    # Empty cache
    cache = InMemoryCache()
    await cache.initialize()
    
    processor = EmbeddingProcessor(encoder, index, cache)
    await processor.initialize()
    
    # Process event
    embedded_event = await processor.process_async(sample_canonical_event)
    
    # Verify encoder WAS called
    encoder.encode_async.assert_called_once()
    
    # Verify cache was updated
    cached = await cache.get(sample_canonical_event.content_hash)
    assert cached == mock_embedding
    
    # Verify Qdrant upsert was called
    index.upsert_async.assert_called_once()
    
    # Verify embedded event
    assert embedded_event.embedding == mock_embedding


@pytest.mark.asyncio
async def test_process_without_cache(sample_canonical_event, mock_embedding):
    """Test processing without cache."""
    # Mock encoder
    encoder = EmbeddingEncoder()
    encoder.encode_async = AsyncMock(return_value=mock_embedding)
    encoder.model_name = "Qwen/Qwen3-Embedding-4B"
    encoder.embedding_dim = 2048
    encoder._initialized = True  # Skip initialization
    
    # Mock index
    index = QdrantIndex()
    index.upsert_async = AsyncMock()
    index.initialize = AsyncMock()  # Mock initialize to prevent real connection
    index._initialized = True  # Mark as initialized
    
    processor = EmbeddingProcessor(encoder, index, cache=None)
    await processor.initialize()
    
    # Process event
    embedded_event = await processor.process_async(sample_canonical_event)
    
    # Verify encoder was called
    encoder.encode_async.assert_called_once()
    
    # Verify Qdrant upsert was called
    index.upsert_async.assert_called_once()


@pytest.mark.asyncio
async def test_batch_processing(sample_canonical_event):
    """Test batch processing."""
    # Create multiple events
    events = [sample_canonical_event] * 5
    
    # Mock encoder
    encoder = EmbeddingEncoder(batch_size=48)
    encoder.encode_batch_async = AsyncMock(
        return_value=[[0.1] * 2048] * 5
    )
    encoder.model_name = "Qwen/Qwen3-Embedding-4B"
    encoder.embedding_dim = 2048
    encoder._initialized = True  # Skip initialization
    
    # Mock index
    index = QdrantIndex()
    index.upsert_async = AsyncMock()
    index.initialize = AsyncMock()  # Mock initialize to prevent real connection
    index._initialized = True  # Mark as initialized
    
    cache = InMemoryCache()
    await cache.initialize()
    
    processor = EmbeddingProcessor(encoder, index, cache, batch_size=5)
    await processor.initialize()
    await processor.start()
    
    # Enqueue events
    for event in events:
        await processor.enqueue(event)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Stop processor
    await processor.stop()
    
    # Verify batch encoding was called
    encoder.encode_batch_async.assert_called_once()
    
    # Verify batch upsert was called
    index.upsert_async.assert_called_once()
    call_args = index.upsert_async.call_args[0][0]
    assert len(call_args) == 5


@pytest.mark.asyncio
async def test_batch_timeout(sample_canonical_event, mock_embedding):
    """Test batch timeout triggers processing."""
    # Mock encoder
    encoder = EmbeddingEncoder()
    encoder.encode_batch_async = AsyncMock(
        return_value=[[0.1] * 2048]
    )
    encoder.model_name = "Qwen/Qwen3-Embedding-4B"
    encoder.embedding_dim = 2048
    
    # Mock index
    index = QdrantIndex()
    index.upsert_async = AsyncMock()
    index.initialize = AsyncMock()  # Mock initialize to prevent real connection
    index._initialized = True  # Mark as initialized
    
    cache = InMemoryCache()
    await cache.initialize()
    
    # Short timeout
    processor = EmbeddingProcessor(
        encoder, index, cache,
        batch_size=10,
        batch_timeout=0.1
    )
    await processor.initialize()
    await processor.start()
    
    # Enqueue one event (won't fill batch)
    await processor.enqueue(sample_canonical_event)
    
    # Wait for timeout
    await asyncio.sleep(0.2)
    
    # Stop processor
    await processor.stop()
    
    # Should have processed due to timeout
    encoder.encode_batch_async.assert_called_once()


@pytest.mark.asyncio
async def test_processor_initialization():
    """Test processor initialization."""
    encoder = EmbeddingEncoder()
    encoder._initialized = True  # Skip initialization
    index = QdrantIndex()
    index.initialize = AsyncMock()  # Mock initialize to prevent real connection
    index._initialized = True  # Mark as initialized
    cache = InMemoryCache()
    
    processor = EmbeddingProcessor(encoder, index, cache)
    
    await processor.initialize()
    
    # All components should be initialized
    assert processor.encoder is not None
    assert processor.index is not None
    assert processor.cache is not None


@pytest.mark.asyncio
async def test_processor_start_stop():
    """Test processor start/stop lifecycle."""
    encoder = EmbeddingEncoder()
    encoder._initialized = True  # Skip initialization
    index = QdrantIndex()
    index.initialize = AsyncMock()  # Mock initialize to prevent real connection
    index._initialized = True  # Mark as initialized
    cache = InMemoryCache()
    
    processor = EmbeddingProcessor(encoder, index, cache)
    await processor.initialize()
    
    # Start
    await processor.start()
    assert processor._running
    assert processor._processing_task is not None
    
    # Stop
    await processor.stop()
    assert not processor._running


@pytest.mark.asyncio
async def test_processor_multiple_start():
    """Test that multiple starts don't create duplicate tasks."""
    encoder = EmbeddingEncoder()
    encoder._initialized = True  # Skip initialization
    index = QdrantIndex()
    index.initialize = AsyncMock()  # Mock initialize to prevent real connection
    index._initialized = True  # Mark as initialized
    cache = InMemoryCache()
    
    processor = EmbeddingProcessor(encoder, index, cache)
    await processor.initialize()
    
    # Start multiple times
    await processor.start()
    task1 = processor._processing_task
    
    await processor.start()
    task2 = processor._processing_task
    
    # Should be same task
    assert task1 == task2
    
    await processor.stop()


@pytest.mark.asyncio
async def test_batch_mixed_cache_hits_misses(sample_canonical_event, mock_embedding):
    """Test batch processing with mixed cache hits and misses."""
    from canonicalization.types import CanonicalEvent
    
    # Create two separate events (not same object reference)
    event1 = sample_canonical_event
    event2 = CanonicalEvent(
        event=sample_canonical_event.event,
        canonical_text=sample_canonical_event.canonical_text,
        content_hash="different-hash",  # Different content hash
        identity_hash=sample_canonical_event.identity_hash  # Same identity
    )
    
    # Mock encoder (only called for cache misses)
    encoder = EmbeddingEncoder()
    encoder.encode_batch_async = AsyncMock(
        return_value=[[0.2] * 2048]  # Only one embedding (for event2)
    )
    encoder.model_name = "Qwen/Qwen3-Embedding-4B"
    encoder.embedding_dim = 2048
    encoder._initialized = True  # Skip initialization
    
    # Mock index
    index = QdrantIndex()
    index.upsert_async = AsyncMock()
    index.initialize = AsyncMock()  # Mock initialize to prevent real connection
    index._initialized = True  # Mark as initialized
    
    # Cache with event1 pre-populated
    cache = InMemoryCache()
    await cache.initialize()
    await cache.set(event1.content_hash, mock_embedding)
    
    processor = EmbeddingProcessor(encoder, index, cache, batch_size=2)
    await processor.initialize()
    await processor.start()
    
    # Enqueue both events
    await processor.enqueue(event1)
    await processor.enqueue(event2)
    
    # Wait for processing
    await asyncio.sleep(0.3)
    
    await processor.stop()
    
    # Should have encoded only event2 (event1 was cache hit)
    encoder.encode_batch_async.assert_called_once()
    call_args = encoder.encode_batch_async.call_args[0][0]
    assert len(call_args) == 1  # Only event2
    
    # Both should be upserted
    index.upsert_async.assert_called_once()
    call_args = index.upsert_async.call_args[0][0]
    assert len(call_args) == 2  # Both events

