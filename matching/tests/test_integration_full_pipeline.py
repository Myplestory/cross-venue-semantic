"""End-to-end integration tests for full matching pipeline.

Tests the complete flow: Retrieval → Cross-Encoder → Reranker
with real Qdrant and real models.
"""

import gc
import pytest
import asyncio
from datetime import datetime, UTC

import torch

from matching.retriever import CandidateRetriever
from matching.reranker import CandidateReranker
from matching.cross_encoder import CrossEncoder
from matching.types import CandidateMatch, VerifiedMatch
from embedding.index import QdrantIndex
from embedding.types import EmbeddedEvent
from discovery.types import VenueType
import config


def _flush_gpu_memory() -> None:
    """
    Force GPU memory cleanup after a model reference has been deleted.

    Must be called AFTER ``del encoder`` in the caller's scope so that
    the reference count drops to zero before GC runs.  Follows the
    canonical PyTorch pattern: del reference → gc.collect() → empty_cache().

    Required on GPUs with limited VRAM (e.g. 8 GB RTX 3070) to avoid
    memory contention when a second model (cross-encoder) runs immediately
    after the first (embedding encoder).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_retrieval_to_rerank(real_qdrant_config, sample_canonical_event, shared_cross_encoder):
    """Test full pipeline: retrieval → cross-encoder → reranker."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    # Setup Qdrant index
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    # Create embedded event
    from embedding.encoder import EmbeddingEncoder
    encoder = EmbeddingEncoder()
    await encoder.initialize()
    
    embedding = await encoder.encode_async(sample_canonical_event.canonical_text)
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    # Free embedding model VRAM before cross-encoder reranking
    del encoder
    _flush_gpu_memory()
    
    # Phase 1: Retrieval
    retriever = CandidateRetriever(index, default_top_k=10, default_score_threshold=0.0)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(embedded_event)
    
    assert len(candidates) > 0
    assert all(isinstance(c, CandidateMatch) for c in candidates)
    
    # Phase 2: Cross-Encoder + Reranker
    # Use shared cross_encoder fixture to avoid redundant model loading
    reranker = CandidateReranker(shared_cross_encoder, top_k=5, score_threshold=0.5)
    await reranker.initialize()
    
    verified_matches = await reranker.rerank_async(sample_canonical_event, candidates)
    
    # Verify results
    assert isinstance(verified_matches, list)
    assert len(verified_matches) <= 5  # Top-K filtering
    
    for match in verified_matches:
        assert isinstance(match, VerifiedMatch)
        assert match.cross_encoder_score >= 0.5
        assert match.match_type in ["full_match", "partial_match", "no_match"]
        assert match.primary_event_score is not None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_cross_venue_matching(real_qdrant_config, sample_canonical_event, shared_cross_encoder):
    """Test full pipeline with cross-venue matching."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    # Setup components
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    from embedding.encoder import EmbeddingEncoder
    encoder = EmbeddingEncoder()
    await encoder.initialize()
    
    embedding = await encoder.encode_async(sample_canonical_event.canonical_text)
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    # Free embedding model VRAM before cross-encoder reranking
    del encoder
    _flush_gpu_memory()
    
    # Retrieve with cross-venue (exclude query venue)
    retriever = CandidateRetriever(index, default_top_k=10, default_score_threshold=0.0)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(
        embedded_event,
        exclude_venue=sample_canonical_event.event.venue
    )
    
    # Verify cross-venue results
    if candidates:
        for candidate in candidates:
            assert candidate.canonical_event.event.venue != sample_canonical_event.event.venue
        
        # Rerank cross-venue matches
        # Use shared cross_encoder fixture
        reranker = CandidateReranker(shared_cross_encoder, score_threshold=0.5)
        await reranker.initialize()
        
        verified = await reranker.rerank_async(sample_canonical_event, candidates)
        
        # Should find some cross-venue matches
        assert isinstance(verified, list)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.asyncio
async def test_full_pipeline_performance(real_qdrant_config, sample_canonical_event, shared_cross_encoder):
    """Test full pipeline performance."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    import time
    
    # Setup
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    from embedding.encoder import EmbeddingEncoder
    encoder = EmbeddingEncoder()
    await encoder.initialize()
    
    embedding = await encoder.encode_async(sample_canonical_event.canonical_text)
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    # Free embedding model VRAM before timed cross-encoder section
    del encoder
    _flush_gpu_memory()
    
    retriever = CandidateRetriever(index, default_top_k=10)
    await retriever.initialize()
    
    # Use shared cross_encoder fixture
    reranker = CandidateReranker(shared_cross_encoder, top_k=5)
    await reranker.initialize()
    
    # Measure full pipeline time
    start = time.time()
    
    candidates = await retriever.retrieve_candidates(embedded_event)
    verified = await reranker.rerank_async(sample_canonical_event, candidates)
    
    elapsed = time.time() - start
    
    # Device-aware performance thresholds (industry standard)
    # Different hardware has different performance characteristics
    if torch.cuda.is_available():
        # CUDA: Fastest, allow headroom for first-run kernel warmup
        max_elapsed = 15.0
        device = "CUDA"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS: Slower than CUDA, expect < 90s for full pipeline
        max_elapsed = 90.0
        device = "MPS"
    else:
        # CPU: Slowest, allow generous threshold for cross-platform compatibility
        # Windows CPU-only PyTorch can be significantly slower than Mac
        max_elapsed = 180.0
        device = "CPU"
    
    assert elapsed < max_elapsed, (
        f"Pipeline took {elapsed:.2f}s, exceeded {max_elapsed}s threshold for {device}"
    )
    
    print(f"\nFull pipeline time: {elapsed:.2f}s (threshold: {max_elapsed}s for {device})")
    print(f"  - Candidates retrieved: {len(candidates)}")
    print(f"  - Verified matches: {len(verified)}")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_verified_match_structure(real_qdrant_config, sample_canonical_event, shared_cross_encoder):
    """Test that VerifiedMatch has complete structure after full pipeline."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    # Setup
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    from embedding.encoder import EmbeddingEncoder
    encoder = EmbeddingEncoder()
    await encoder.initialize()
    
    embedding = await encoder.encode_async(sample_canonical_event.canonical_text)
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    # Free embedding model VRAM before cross-encoder reranking
    del encoder
    _flush_gpu_memory()
    
    retriever = CandidateRetriever(index, default_top_k=5)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(embedded_event)
    
    if not candidates:
        pytest.skip("No candidates found in Qdrant")
    
    # Use shared cross_encoder fixture
    reranker = CandidateReranker(shared_cross_encoder, score_threshold=0.0)  # Low threshold to get results
    await reranker.initialize()
    
    verified = await reranker.rerank_async(sample_canonical_event, candidates)
    
    if verified:
        match = verified[0]
        
        # Verify complete structure
        assert match.candidate_match is not None
        assert match.cross_encoder_score is not None
        assert match.match_type in ["full_match", "partial_match", "no_match"]
        assert match.nli_scores is not None
        assert match.primary_event_score is not None
        assert match.verified_at is not None
        assert match.verification_metadata is not None
        
        # Verify candidate match structure
        assert match.candidate_match.canonical_event is not None
        assert match.candidate_match.similarity_score is not None
        assert match.candidate_match.embedding is not None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_score_consistency(real_qdrant_config, sample_canonical_event, shared_cross_encoder):
    """Test that scores are consistent across pipeline stages."""
    if not real_qdrant_config["api_key"]:
        pytest.skip("QDRANT_API_KEY not set in .env")
    
    # Setup
    index = QdrantIndex(
        url=real_qdrant_config["url"],
        api_key=real_qdrant_config["api_key"],
        collection_name=real_qdrant_config["collection_name"],
        vector_size=real_qdrant_config["vector_size"]
    )
    
    from embedding.encoder import EmbeddingEncoder
    encoder = EmbeddingEncoder()
    await encoder.initialize()
    
    embedding = await encoder.encode_async(sample_canonical_event.canonical_text)
    embedded_event = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=embedding,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    # Free embedding model VRAM before cross-encoder reranking
    del encoder
    _flush_gpu_memory()
    
    retriever = CandidateRetriever(index, default_top_k=5)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(embedded_event)
    
    if not candidates:
        pytest.skip("No candidates found in Qdrant")
    
    # Store original similarity scores
    original_scores = {c.canonical_event.identity_hash: c.similarity_score for c in candidates}
    
    # Use shared cross_encoder fixture
    reranker = CandidateReranker(shared_cross_encoder, score_threshold=0.0)
    await reranker.initialize()
    
    verified = await reranker.rerank_async(sample_canonical_event, candidates)
    
    # Verify that original similarity scores are preserved
    for match in verified:
        original_score = original_scores.get(match.candidate_match.canonical_event.identity_hash)
        assert original_score is not None
        assert match.candidate_match.similarity_score == original_score
        
        # Cross-encoder score should be different (re-ranked)
        assert match.cross_encoder_score != original_score

