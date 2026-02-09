"""End-to-end integration tests for embedding phase."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from embedding.processor import EmbeddingProcessor
from embedding.encoder import EmbeddingEncoder
from embedding.index import QdrantIndex
from embedding.cache.in_memory import InMemoryCache
from embedding.types import EmbeddedEvent
import config


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_embedding_flow(real_qdrant_config, sample_canonical_event):
    """Test full embedding pipeline end-to-end."""
    # Skip if no API key configured
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    # Use real Qdrant from .env
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    # Mock encoder (to avoid model download in tests)
    encoder = EmbeddingEncoder(
        model_name=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    # Mock the model to avoid download
    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        return_value=np.array([[0.1] * config.EMBEDDING_DIM])
    )
    mock_model.eval = MagicMock()
    
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        await encoder.initialize()
    
    # Real cache
    cache = InMemoryCache(max_size=config.EMBEDDING_CACHE_MAX_SIZE)
    
    processor = EmbeddingProcessor(encoder, index, cache)
    await processor.initialize()
    
    # Process event
    embedded_event = await processor.process_async(sample_canonical_event)
    
    # Verify embedded event
    assert embedded_event is not None
    assert len(embedded_event.embedding) == config.EMBEDDING_DIM
    assert embedded_event.canonical_event == sample_canonical_event
    
    # Verify stored in Qdrant
    result = await index.get_by_identity_hash(
        sample_canonical_event.identity_hash
    )
    assert result is not None
    # Point ID is UUID, but identity_hash is stored in payload
    assert result["payload"]["identity_hash"] == sample_canonical_event.identity_hash
    
    # Verify cached
    cached = await cache.get(sample_canonical_event.content_hash)
    assert cached == embedded_event.embedding


@pytest.mark.integration
@pytest.mark.asyncio
async def test_similarity_search(real_qdrant_config, sample_canonical_event):
    """Test similarity search with real Qdrant."""
    # Skip if no API key configured
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    await index.initialize()
    
    # Create and upsert embedded event
    mock_embedding = [0.1] * config.EMBEDDING_DIM
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    await index.upsert_async([embedded_event])
    
    # Search for similar
    results = await index.search_async(
        query_vector=mock_embedding,
        top_k=10,
        score_threshold=0.7
    )
    
    assert len(results) > 0
    
    # Find our inserted event in the results (may not be first due to other test data)
    our_result = None
    for result in results:
        if result["payload"]["identity_hash"] == sample_canonical_event.identity_hash:
            our_result = result
            break
    
    # Verify our event was found and has good similarity score
    assert our_result is not None, f"Expected event with identity_hash {sample_canonical_event.identity_hash} not found in results"
    assert our_result["score"] >= 0.7


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_integration(real_qdrant_config, sample_canonical_event):
    """Test cache integration with real Qdrant."""
    # Skip if no API key configured
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    # Mock encoder
    encoder = EmbeddingEncoder(
        model_name=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        return_value=np.array([[0.1] * config.EMBEDDING_DIM])
    )
    mock_model.eval = MagicMock()
    
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        await encoder.initialize()
    
    cache = InMemoryCache(max_size=config.EMBEDDING_CACHE_MAX_SIZE)
    
    processor = EmbeddingProcessor(encoder, index, cache)
    await processor.initialize()
    
    # First process (cache miss)
    embedded_event1 = await processor.process_async(sample_canonical_event)
    
    # Verify encoder was called
    assert mock_model.encode.called
    
    # Reset mock
    mock_model.encode.reset_mock()
    
    # Second process (cache hit)
    embedded_event2 = await processor.process_async(sample_canonical_event)
    
    # Verify encoder was NOT called (cache hit)
    assert not mock_model.encode.called
    
    # Both should have same embedding
    assert embedded_event1.embedding == embedded_event2.embedding


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_processing_integration(real_qdrant_config, sample_canonical_event):
    """Test batch processing with real Qdrant."""
    # Skip if no API key configured
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    # Mock encoder
    encoder = EmbeddingEncoder(
        model_name=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM,
        batch_size=config.EMBEDDING_BATCH_SIZE
    )
    
    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        return_value=np.array([[0.1] * config.EMBEDDING_DIM] * 3)
    )
    mock_model.eval = MagicMock()
    
    with patch('embedding.encoder.SentenceTransformer', return_value=mock_model):
        await encoder.initialize()
    
    cache = InMemoryCache(max_size=config.EMBEDDING_CACHE_MAX_SIZE)
    
    processor = EmbeddingProcessor(encoder, index, cache, batch_size=3)
    await processor.initialize()
    await processor.start()
    
    # Enqueue multiple events - create new event objects for each
    from canonicalization.types import CanonicalEvent
    events = []
    for i in range(3):
        # Create a new CanonicalEvent with unique identity_hash
        event = CanonicalEvent(
            event=sample_canonical_event.event,
            canonical_text=sample_canonical_event.canonical_text,
            content_hash=f"{sample_canonical_event.content_hash}-{i}",  # Unique content hash too
            identity_hash=f"test-batch-{i}"  # Unique identity hash
        )
        events.append(event)
        await processor.enqueue(event)
    
    # Wait for processing (batch timeout is 2.0s, so wait a bit longer)
    import asyncio
    await asyncio.sleep(3.0)  # Give enough time for batch processing
    
    await processor.stop()
    
    # Verify batch encoding was called
    assert mock_model.encode.called
    
    # Verify all events were upserted
    for i, event in enumerate(events):
        result = await index.get_by_identity_hash(event.identity_hash)
        assert result is not None, f"Event {i} with identity_hash {event.identity_hash} not found in Qdrant"

