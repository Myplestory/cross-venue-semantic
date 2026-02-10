"""Unit tests for CandidateReranker with mocked CrossEncoder."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from matching.reranker import CandidateReranker
from matching.types import CandidateMatch, VerifiedMatch
from matching.cross_encoder import CrossEncoder


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reranker_initialization_defaults(mock_cross_encoder):
    """Test reranker initialization with default parameters."""
    reranker = CandidateReranker(mock_cross_encoder)
    
    assert reranker.cross_encoder == mock_cross_encoder
    assert reranker.top_k == 10
    assert reranker.score_threshold == 0.7
    assert reranker.primary_weight == 0.7
    assert reranker.secondary_weight == 0.3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reranker_initialization_custom_params(mock_cross_encoder):
    """Test reranker with custom parameters."""
    reranker = CandidateReranker(
        mock_cross_encoder,
        top_k=20,
        score_threshold=0.8,
        primary_weight=0.8,
        secondary_weight=0.2
    )
    
    assert reranker.top_k == 20
    assert reranker.score_threshold == 0.8
    assert reranker.primary_weight == 0.8
    assert reranker.secondary_weight == 0.2


@pytest.mark.unit
def test_reranker_initialization_weight_validation(mock_cross_encoder):
    """Test that weights must sum to 1.0."""
    # Valid weights
    reranker = CandidateReranker(mock_cross_encoder, primary_weight=0.7, secondary_weight=0.3)
    assert reranker.primary_weight + reranker.secondary_weight == 1.0
    
    # Invalid weights (should raise ValueError)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        CandidateReranker(mock_cross_encoder, primary_weight=0.7, secondary_weight=0.4)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reranker_initialize(mock_cross_encoder):
    """Test that initialize calls cross_encoder.initialize()."""
    mock_cross_encoder.initialize = AsyncMock()
    
    reranker = CandidateReranker(mock_cross_encoder)
    await reranker.initialize()
    
    mock_cross_encoder.initialize.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_empty_candidates(mock_cross_encoder, sample_canonical_event):
    """Test reranking with empty candidate list."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=[])
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.8, "full_match"))
    
    reranker = CandidateReranker(mock_cross_encoder)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, [])
    
    assert results == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_single_candidate(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test reranking with single candidate."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Query primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=["Clause 1"])
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.9,
        "neutral": 0.05,
        "contradiction": 0.05
    }])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.9, "full_match"))
    mock_cross_encoder.score_secondary_clauses_async = AsyncMock(return_value=0.85)
    
    reranker = CandidateReranker(mock_cross_encoder, score_threshold=0.7)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, [sample_candidate_matches[0]])
    
    assert len(results) == 1
    assert isinstance(results[0], VerifiedMatch)
    assert results[0].cross_encoder_score >= 0.7
    assert results[0].match_type == "full_match"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_multiple_candidates(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test reranking with multiple candidates."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Query primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=["Clause 1"])
    
    # Return different scores for each candidate
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[
        {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},  # High score
        {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1},     # Medium score
        {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2},    # Low score
    ])
    
    # Map to confidence scores (decreasing)
    mock_cross_encoder.map_nli_to_confidence = MagicMock(side_effect=[
        (0.9, "full_match"),
        (0.7, "partial_match"),
        (0.5, "no_match"),
    ])
    
    mock_cross_encoder.score_secondary_clauses_async = AsyncMock(return_value=0.8)
    
    reranker = CandidateReranker(mock_cross_encoder, score_threshold=0.6, top_k=2)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, sample_candidate_matches[:3])
    
    assert len(results) == 2  # Top 2 only
    # Should be sorted by confidence (highest first)
    assert results[0].cross_encoder_score >= results[1].cross_encoder_score


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_score_ordering(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test that results are sorted by confidence (highest first)."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Query primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=[])
    
    # Return scores in random order
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[
        {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2},  # Lowest
        {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},  # Highest
        {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1},   # Middle
    ])
    
    mock_cross_encoder.map_nli_to_confidence = MagicMock(side_effect=[
        (0.5, "no_match"),
        (0.9, "full_match"),
        (0.7, "partial_match"),
    ])
    
    reranker = CandidateReranker(mock_cross_encoder, score_threshold=0.0)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, sample_candidate_matches[:3])
    
    # Should be sorted descending by confidence
    scores = [r.cross_encoder_score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_top_k_filtering(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test top-k filtering."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Query primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=[])
    
    # Return high scores for all
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[
        {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05} for _ in range(5)
    ])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.9, "full_match"))
    
    reranker = CandidateReranker(mock_cross_encoder, top_k=3, score_threshold=0.0)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, sample_candidate_matches)
    
    assert len(results) == 3  # Top 3 only


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_score_threshold_filtering(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test score threshold filtering."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Query primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=[])
    
    # Return mixed scores
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[
        {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},  # Above threshold
        {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2},    # Below threshold
        {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1},   # Above threshold
    ])
    
    mock_cross_encoder.map_nli_to_confidence = MagicMock(side_effect=[
        (0.9, "full_match"),
        (0.5, "no_match"),
        (0.8, "full_match"),
    ])
    
    reranker = CandidateReranker(mock_cross_encoder, score_threshold=0.7)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, sample_candidate_matches[:3])
    
    # Only results above threshold
    assert all(r.cross_encoder_score >= 0.7 for r in results)
    assert len(results) == 2  # Two above threshold


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_primary_event_extraction(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test that primary events are extracted correctly."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Extracted primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=[])
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.9,
        "neutral": 0.05,
        "contradiction": 0.05
    }])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.9, "full_match"))
    
    reranker = CandidateReranker(mock_cross_encoder)
    await reranker.initialize()
    
    await reranker.rerank_async(sample_canonical_event, [sample_candidate_matches[0]])
    
    # Should extract primary event from query and candidates
    calls = mock_cross_encoder.extract_primary_event.call_args_list
    assert len(calls) >= 2  # Query + at least one candidate
    
    # Verify query canonical text was extracted (should be first call)
    query_calls = [call[0][0] for call in calls if call[0][0] == sample_canonical_event.canonical_text]
    assert len(query_calls) > 0, "Query canonical text should be extracted"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_with_secondary_clauses(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test reranking when both markets have secondary clauses."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=["Clause 1", "Clause 2"])
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.8,
        "neutral": 0.15,
        "contradiction": 0.05
    }])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.8, "full_match"))
    mock_cross_encoder.score_secondary_clauses_async = AsyncMock(return_value=0.75)
    
    reranker = CandidateReranker(mock_cross_encoder, primary_weight=0.7, secondary_weight=0.3)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, [sample_candidate_matches[0]])
    
    assert len(results) == 1
    # Combined score should be weighted: 0.7 * 0.8 + 0.3 * 0.75 = 0.785
    assert results[0].secondary_clause_score == 0.75
    assert results[0].cross_encoder_score > 0.7  # Should be combined score


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_without_secondary_clauses(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test reranking when no secondary clauses exist."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=[])  # No clauses
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.8,
        "neutral": 0.15,
        "contradiction": 0.05
    }])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.8, "full_match"))
    
    reranker = CandidateReranker(mock_cross_encoder)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, [sample_candidate_matches[0]])
    
    assert len(results) == 1
    # Should use primary score only (no secondary)
    assert results[0].secondary_clause_score is None
    assert results[0].cross_encoder_score == 0.8  # Primary score only


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_early_stopping_low_primary(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test early stopping when primary score is too low."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=["Clause 1"])
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.2,  # Very low
        "neutral": 0.3,
        "contradiction": 0.5
    }])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.2, "no_match"))
    
    reranker = CandidateReranker(mock_cross_encoder, score_threshold=0.7)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, [sample_candidate_matches[0]])
    
    # Should skip secondary evaluation (primary too low)
    # Should not call score_secondary_clauses_async
    # Result should be filtered out (below threshold)
    assert len(results) == 0  # Filtered out by threshold


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_full_match_classification(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test full_match classification."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=[])
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.9,
        "neutral": 0.05,
        "contradiction": 0.05
    }])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.9, "full_match"))
    mock_cross_encoder.entailment_threshold = 0.7
    
    reranker = CandidateReranker(mock_cross_encoder, score_threshold=0.7)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, [sample_candidate_matches[0]])
    
    assert len(results) == 1
    assert results[0].match_type == "full_match"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_partial_match_classification(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test partial_match classification."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=[])
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.5,
        "neutral": 0.4,
        "contradiction": 0.1
    }])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.6, "partial_match"))
    
    reranker = CandidateReranker(mock_cross_encoder, score_threshold=0.5)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, [sample_candidate_matches[0]])
    
    assert len(results) == 1
    assert results[0].match_type == "partial_match"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_preserves_candidate_metadata(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test that candidate metadata is preserved in VerifiedMatch."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=[])
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.9,
        "neutral": 0.05,
        "contradiction": 0.05
    }])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.9, "full_match"))
    
    reranker = CandidateReranker(mock_cross_encoder)
    await reranker.initialize()
    
    candidate = sample_candidate_matches[0]
    results = await reranker.rerank_async(sample_canonical_event, [candidate])
    
    assert len(results) == 1
    verified = results[0]
    assert verified.candidate_match == candidate
    assert verified.candidate_match.canonical_event == candidate.canonical_event
    assert verified.candidate_match.similarity_score == candidate.similarity_score


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rerank_async_verified_match_structure(mock_cross_encoder, sample_canonical_event, sample_candidate_matches):
    """Test that VerifiedMatch has correct structure."""
    mock_cross_encoder.initialize = AsyncMock()
    mock_cross_encoder.extract_primary_event = MagicMock(return_value="Primary event")
    mock_cross_encoder.extract_secondary_clauses = MagicMock(return_value=["Clause 1"])
    mock_cross_encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.8,
        "neutral": 0.15,
        "contradiction": 0.05
    }])
    mock_cross_encoder.map_nli_to_confidence = MagicMock(return_value=(0.8, "full_match"))
    mock_cross_encoder.score_secondary_clauses_async = AsyncMock(return_value=0.75)
    
    reranker = CandidateReranker(mock_cross_encoder)
    await reranker.initialize()
    
    results = await reranker.rerank_async(sample_canonical_event, [sample_candidate_matches[0]])
    
    assert len(results) == 1
    verified = results[0]
    
    # Verify structure
    assert isinstance(verified, VerifiedMatch)
    assert verified.candidate_match is not None
    assert verified.cross_encoder_score is not None
    assert verified.match_type in ["full_match", "partial_match", "no_match"]
    assert verified.nli_scores is not None
    assert verified.primary_event_score is not None
    assert verified.secondary_clause_score is not None
    assert verified.verified_at is not None
    assert verified.verification_metadata is not None

