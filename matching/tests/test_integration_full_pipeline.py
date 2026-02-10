"""End-to-end integration tests for full matching pipeline.

Tests the complete flow: Retrieval → Cross-Encoder → Reranker
with real Qdrant and real models.
"""

import pytest
import asyncio
from datetime import datetime, UTC

from matching.retriever import CandidateRetriever
from matching.reranker import CandidateReranker
from matching.cross_encoder import CrossEncoder
from matching.types import CandidateMatch, VerifiedMatch
from embedding.index import QdrantIndex
from embedding.types import EmbeddedEvent
from discovery.types import VenueType
import config


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_retrieval_to_rerank(real_qdrant_config, sample_canonical_event):
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
    
    # Phase 1: Retrieval
    retriever = CandidateRetriever(index, default_top_k=10, default_score_threshold=0.0)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(embedded_event)
    
    assert len(candidates) > 0
    assert all(isinstance(c, CandidateMatch) for c in candidates)
    
    # Phase 2: Cross-Encoder + Reranker
    cross_encoder = CrossEncoder()
    await cross_encoder.initialize()
    
    reranker = CandidateReranker(cross_encoder, top_k=5, score_threshold=0.5)
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
async def test_full_pipeline_cross_venue_matching(real_qdrant_config, sample_canonical_event):
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
        cross_encoder = CrossEncoder()
        await cross_encoder.initialize()
        
        reranker = CandidateReranker(cross_encoder, score_threshold=0.5)
        await reranker.initialize()
        
        verified = await reranker.rerank_async(sample_canonical_event, candidates)
        
        # Should find some cross-venue matches
        assert isinstance(verified, list)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.asyncio
async def test_full_pipeline_performance(real_qdrant_config, sample_canonical_event):
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
    
    retriever = CandidateRetriever(index, default_top_k=10)
    await retriever.initialize()
    
    cross_encoder = CrossEncoder()
    await cross_encoder.initialize()
    
    reranker = CandidateReranker(cross_encoder, top_k=5)
    await reranker.initialize()
    
    # Measure full pipeline time
    start = time.time()
    
    candidates = await retriever.retrieve_candidates(embedded_event)
    verified = await reranker.rerank_async(sample_canonical_event, candidates)
    
    elapsed = time.time() - start
    
    # Performance targets (adjust based on your hardware)
    # Full pipeline should complete in reasonable time
    assert elapsed < 10.0  # 10 seconds for full pipeline (retrieval + reranking)
    
    print(f"\nFull pipeline time: {elapsed:.2f}s")
    print(f"  - Candidates retrieved: {len(candidates)}")
    print(f"  - Verified matches: {len(verified)}")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_verified_match_structure(real_qdrant_config, sample_canonical_event):
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
    
    retriever = CandidateRetriever(index, default_top_k=5)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(embedded_event)
    
    if not candidates:
        pytest.skip("No candidates found in Qdrant")
    
    cross_encoder = CrossEncoder()
    await cross_encoder.initialize()
    
    reranker = CandidateReranker(cross_encoder, score_threshold=0.0)  # Low threshold to get results
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
async def test_full_pipeline_score_consistency(real_qdrant_config, sample_canonical_event):
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
    
    retriever = CandidateRetriever(index, default_top_k=5)
    await retriever.initialize()
    
    candidates = await retriever.retrieve_candidates(embedded_event)
    
    if not candidates:
        pytest.skip("No candidates found in Qdrant")
    
    # Store original similarity scores
    original_scores = {c.canonical_event.identity_hash: c.similarity_score for c in candidates}
    
    cross_encoder = CrossEncoder()
    await cross_encoder.initialize()
    
    reranker = CandidateReranker(cross_encoder, score_threshold=0.0)
    await reranker.initialize()
    
    verified = await reranker.rerank_async(sample_canonical_event, candidates)
    
    # Verify that original similarity scores are preserved
    for match in verified:
        original_score = original_scores.get(match.candidate_match.canonical_event.identity_hash)
        assert original_score is not None
        assert match.candidate_match.similarity_score == original_score
        
        # Cross-encoder score should be different (re-ranked)
        assert match.cross_encoder_score != original_score

