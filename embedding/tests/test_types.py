"""Tests for EmbeddedEvent dataclass."""

import pytest
from datetime import datetime, UTC

from embedding.types import EmbeddedEvent
from canonicalization.types import CanonicalEvent
from discovery.types import MarketEvent, VenueType, EventType


def test_embedded_event_creation(sample_canonical_event, mock_embedding):
    """Test creating EmbeddedEvent with valid data."""
    embedded = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dim=2048
    )
    
    assert embedded.canonical_event == sample_canonical_event
    assert len(embedded.embedding) == 2048
    assert embedded.embedding_model == "Qwen/Qwen3-Embedding-4B"
    assert embedded.embedding_dim == 2048
    assert embedded.created_at is not None
    assert isinstance(embedded.created_at, datetime)


def test_embedded_event_default_timestamp(sample_canonical_event, mock_embedding):
    """Test that created_at is set automatically if not provided."""
    embedded = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dim=2048
    )
    
    assert embedded.created_at is not None
    assert isinstance(embedded.created_at, datetime)
    # Should be recent (within last second)
    assert (datetime.now(UTC) - embedded.created_at).total_seconds() < 1


def test_embedded_event_custom_timestamp(sample_canonical_event, mock_embedding):
    """Test that custom timestamp is preserved."""
    custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    
    embedded = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dim=2048,
        created_at=custom_time
    )
    
    assert embedded.created_at == custom_time


def test_embedded_event_dimension_validation(sample_canonical_event):
    """Test that dimension mismatch raises ValueError."""
    wrong_dim_embedding = [0.1] * 512  # Wrong dimension
    
    with pytest.raises(ValueError, match="dimension mismatch"):
        EmbeddedEvent(
            canonical_event=sample_canonical_event,
            embedding=wrong_dim_embedding,
            embedding_model="Qwen/Qwen3-Embedding-4B",
            embedding_dim=2048  # Expects 2048, got 512
        )


def test_embedded_event_different_dimensions(sample_canonical_event):
    """Test EmbeddedEvent with different embedding dimensions."""
    # Test with 32-dim (MRL minimum)
    embedding_32 = [0.1] * 32
    embedded_32 = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=embedding_32,
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dim=32
    )
    assert len(embedded_32.embedding) == 32
    
    # Test with 2560-dim (MRL maximum)
    embedding_2560 = [0.1] * 2560
    embedded_2560 = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=embedding_2560,
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dim=2560
    )
    assert len(embedded_2560.embedding) == 2560


def test_embedded_event_embedding_immutability(sample_canonical_event, mock_embedding):
    """Test that embedding list is stored correctly."""
    embedded = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dim=2048
    )
    
    # Modify original list
    mock_embedding[0] = 999.0
    
    # Embedded event should have original values
    assert embedded.embedding[0] == 0.1


def test_embedded_event_all_fields(sample_canonical_event, mock_embedding):
    """Test that all fields are accessible."""
    embedded = EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dim=2048
    )
    
    assert embedded.canonical_event is not None
    assert embedded.embedding is not None
    assert embedded.embedding_model is not None
    assert embedded.embedding_dim is not None
    assert embedded.created_at is not None

