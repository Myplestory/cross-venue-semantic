"""Pytest configuration for embedding tests."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import numpy as np

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_market_event():
    """Create sample MarketEvent for testing."""
    from discovery.types import MarketEvent, VenueType, EventType, OutcomeSpec
    from datetime import datetime, UTC
    
    return MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="test-market-123",
        event_type=EventType.CREATED,
        title="Will Bitcoin reach $100k by 2025?",
        description="Test market for embedding",
        outcomes=[
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No")
        ],
        received_at=datetime.now(UTC)
    )


@pytest.fixture
def sample_canonical_event(sample_market_event):
    """Create sample CanonicalEvent for testing."""
    from canonicalization.types import CanonicalEvent
    from canonicalization.text_builder import get_builder
    
    # Build canonical text using the actual text builder to match real format
    builder = get_builder(sample_market_event.venue)
    canonical_text = builder.build(sample_market_event)
    
    return CanonicalEvent(
        event=sample_market_event,
        canonical_text=canonical_text,
        content_hash="abc123def456",
        identity_hash="xyz789uvw012"
    )


@pytest.fixture
def mock_embedding():
    """Mock 2048-dimensional embedding vector."""
    return [0.1] * 2048


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer model."""
    model = MagicMock()
    # encode returns 2D array for batch, 1D for single (sentence-transformers behavior)
    # For single text, it returns shape (1, 2048), which becomes [0.1] * 2048 after tolist()[0]
    model.encode = MagicMock(
        return_value=np.array([[0.1] * 2048])  # Shape: (1, 2048) for single text
    )
    model.eval = MagicMock()
    return model


@pytest.fixture
def mock_qdrant_client():
    """Mock AsyncQdrantClient."""
    client = AsyncMock()
    client.get_collections = AsyncMock(
        return_value=MagicMock(collections=[])
    )
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()  # For identity_hash index
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.query_points = AsyncMock(return_value=MagicMock(points=[]))  # For async client fallback
    client.query = AsyncMock(return_value=[])  # Alternative method name
    client.retrieve = AsyncMock(return_value=[])
    client.scroll = AsyncMock(return_value=([], None))  # Returns tuple
    return client


@pytest.fixture
def real_qdrant_config():
    """Get real Qdrant config from .env (for integration tests)."""
    import config
    return {
        "url": config.QDRANT_URL,
        "api_key": config.QDRANT_API_KEY,
        "collection_name": f"{config.QDRANT_COLLECTION_NAME}_test",
        "vector_size": config.QDRANT_VECTOR_SIZE
    }

