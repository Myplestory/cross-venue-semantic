"""Integration tests for CandidateRetriever with real Qdrant."""

import pytest
import asyncio
from datetime import datetime, UTC

from matching.retriever import CandidateRetriever
from matching.types import CandidateMatch
from embedding.index import QdrantIndex
from embedding.types import EmbeddedEvent
from discovery.types import VenueType
import config


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_candidates_real_qdrant(real_qdrant_config, sample_canonical_event):
    """Test retrieval with real Qdrant."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    mock_embedding = [0.1] * config.EMBEDDING_DIM
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    retriever = CandidateRetriever(index)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(
        embedded_event,
        top_k=10,
        score_threshold=0.0
    )
    
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert 0.0 <= candidate.similarity_score <= 1.0
        assert candidate.canonical_event is not None
        assert candidate.retrieved_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_candidates_cross_venue_real(real_qdrant_config, sample_canonical_event):
    """Test cross-venue matching with real Qdrant."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    mock_embedding = [0.1] * config.EMBEDDING_DIM
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    retriever = CandidateRetriever(index)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(
        embedded_event,
        top_k=10,
        score_threshold=0.0,
        exclude_venue=VenueType.KALSHI
    )
    
    for candidate in candidates:
        assert candidate.canonical_event.event.venue != VenueType.KALSHI


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_candidates_score_threshold_real(real_qdrant_config, sample_canonical_event):
    """Test score threshold filtering with real Qdrant."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    mock_embedding = [0.1] * config.EMBEDDING_DIM
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    retriever = CandidateRetriever(index)
    await retriever.initialize()
    
    low_threshold = await retriever.retrieve_candidates(
        embedded_event,
        top_k=10,
        score_threshold=0.0
    )
    
    high_threshold = await retriever.retrieve_candidates(
        embedded_event,
        top_k=10,
        score_threshold=0.9
    )
    
    assert len(low_threshold) >= len(high_threshold)
    for candidate in high_threshold:
        assert candidate.similarity_score >= 0.9


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_candidates_top_k_real(real_qdrant_config, sample_canonical_event):
    """Test top-K limiting with real Qdrant."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    mock_embedding = [0.1] * config.EMBEDDING_DIM
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    retriever = CandidateRetriever(index)
    await retriever.initialize()
    
    top_5 = await retriever.retrieve_candidates(
        embedded_event,
        top_k=5,
        score_threshold=0.0
    )
    
    top_20 = await retriever.retrieve_candidates(
        embedded_event,
        top_k=20,
        score_threshold=0.0
    )
    
    assert len(top_5) <= 5
    assert len(top_20) <= 20
    assert len(top_20) >= len(top_5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_candidates_self_exclusion_real(real_qdrant_config, sample_canonical_event):
    """Test self-exclusion with real Qdrant."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    mock_embedding = [0.1] * config.EMBEDDING_DIM
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    retriever = CandidateRetriever(index)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(
        embedded_event,
        top_k=10,
        score_threshold=0.0
    )
    
    for candidate in candidates:
        assert candidate.canonical_event.identity_hash != sample_canonical_event.identity_hash


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_candidates_multiple_queries(real_qdrant_config, sample_canonical_event):
    """Test multiple sequential queries work correctly."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    mock_embedding = [0.1] * config.EMBEDDING_DIM
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    retriever = CandidateRetriever(index)
    await retriever.initialize()
    
    results1 = await retriever.retrieve_candidates(embedded_event, top_k=5)
    results2 = await retriever.retrieve_candidates(embedded_event, top_k=5)
    
    assert isinstance(results1, list)
    assert isinstance(results2, list)
    assert len(results1) == len(results2)


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.asyncio
async def test_retrieve_candidates_performance(real_qdrant_config, sample_canonical_event):
    """Test retrieval performance meets latency targets."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    mock_embedding = [0.1] * config.EMBEDDING_DIM
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    retriever = CandidateRetriever(index)
    await retriever.initialize()
    
    latencies = []
    for _ in range(10):
        start = asyncio.get_event_loop().time()
        await retriever.retrieve_candidates(embedded_event, top_k=10)
        end = asyncio.get_event_loop().time()
        latencies.append((end - start) * 1000)
    
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    # Performance target: P95 latency < 500ms for real Qdrant cloud instance
    # Accounts for network latency, cloud infrastructure, DNS resolution, and jitter
    assert p95_latency < 500, f"P95 latency {p95_latency}ms exceeds 500ms target"

