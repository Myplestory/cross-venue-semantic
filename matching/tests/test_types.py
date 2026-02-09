"""Tests for CandidateMatch dataclass."""

import pytest
from datetime import datetime, UTC

from matching.types import CandidateMatch
from canonicalization.types import CanonicalEvent
from discovery.types import MarketEvent, VenueType, EventType


def test_candidate_match_creation(sample_canonical_event, mock_embedding):
    """Test creating CandidateMatch with valid data."""
    candidate = CandidateMatch(
        canonical_event=sample_canonical_event,
        similarity_score=0.95,
        embedding=mock_embedding
    )
    
    assert candidate.canonical_event == sample_canonical_event
    assert candidate.similarity_score == 0.95
    assert len(candidate.embedding) == 2048
    assert candidate.retrieved_at is not None
    assert isinstance(candidate.retrieved_at, datetime)


def test_candidate_match_default_timestamp(sample_canonical_event, mock_embedding):
    """Test that retrieved_at is set automatically if not provided."""
    candidate = CandidateMatch(
        canonical_event=sample_canonical_event,
        similarity_score=0.85,
        embedding=mock_embedding
    )
    
    assert candidate.retrieved_at is not None
    assert isinstance(candidate.retrieved_at, datetime)
    assert (datetime.now(UTC) - candidate.retrieved_at).total_seconds() < 1


def test_candidate_match_custom_timestamp(sample_canonical_event, mock_embedding):
    """Test that custom timestamp is preserved."""
    custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    
    candidate = CandidateMatch(
        canonical_event=sample_canonical_event,
        similarity_score=0.90,
        embedding=mock_embedding,
        retrieved_at=custom_time
    )
    
    assert candidate.retrieved_at == custom_time


def test_candidate_match_similarity_score_validation(sample_canonical_event, mock_embedding):
    """Test that valid similarity scores work."""
    candidate1 = CandidateMatch(
        canonical_event=sample_canonical_event,
        similarity_score=0.0,
        embedding=mock_embedding
    )
    assert candidate1.similarity_score == 0.0
    
    candidate2 = CandidateMatch(
        canonical_event=sample_canonical_event,
        similarity_score=0.5,
        embedding=mock_embedding
    )
    assert candidate2.similarity_score == 0.5
    
    candidate3 = CandidateMatch(
        canonical_event=sample_canonical_event,
        similarity_score=1.0,
        embedding=mock_embedding
    )
    assert candidate3.similarity_score == 1.0


def test_candidate_match_similarity_score_too_high(sample_canonical_event, mock_embedding):
    """Test that similarity score >1.0 raises ValueError."""
    with pytest.raises(ValueError, match="Similarity score must be between 0.0 and 1.0"):
        CandidateMatch(
            canonical_event=sample_canonical_event,
            similarity_score=1.5,
            embedding=mock_embedding
        )


def test_candidate_match_similarity_score_negative(sample_canonical_event, mock_embedding):
    """Test that similarity score <0.0 raises ValueError."""
    with pytest.raises(ValueError, match="Similarity score must be between 0.0 and 1.0"):
        CandidateMatch(
            canonical_event=sample_canonical_event,
            similarity_score=-0.1,
            embedding=mock_embedding
        )


def test_candidate_match_embedding_immutability(sample_canonical_event, mock_embedding):
    """Test that embedding list is stored correctly."""
    candidate = CandidateMatch(
        canonical_event=sample_canonical_event,
        similarity_score=0.88,
        embedding=mock_embedding
    )
    
    mock_embedding[0] = 999.0
    
    assert candidate.embedding[0] == 0.1


def test_candidate_match_default_metadata(sample_canonical_event, mock_embedding):
    """Test that default metadata is empty dict if not provided."""
    candidate = CandidateMatch(
        canonical_event=sample_canonical_event,
        similarity_score=0.92,
        embedding=mock_embedding
    )
    
    assert candidate.retrieval_metadata == {}
    
    custom_metadata = {"key": "value"}
    candidate2 = CandidateMatch(
        canonical_event=sample_canonical_event,
        similarity_score=0.93,
        embedding=mock_embedding,
        retrieval_metadata=custom_metadata
    )
    
    assert candidate2.retrieval_metadata == custom_metadata

