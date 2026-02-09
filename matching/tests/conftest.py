"""Pytest configuration for matching tests."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

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
        description="Test market for retrieval",
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
def sample_embedded_event(sample_canonical_event, mock_embedding):
    """Create sample EmbeddedEvent for retrieval testing."""
    from embedding.types import EmbeddedEvent
    
    return EmbeddedEvent(
        canonical_event=sample_canonical_event,
        embedding=mock_embedding,
        embedding_model="Qwen/Qwen3-Embedding-4B",
        embedding_dim=2048
    )


@pytest.fixture
def mock_qdrant_index():
    """Mock QdrantIndex for unit tests."""
    from embedding.index import QdrantIndex
    from unittest.mock import AsyncMock
    
    index = QdrantIndex()
    index.search_async = AsyncMock(return_value=[])
    index.initialize = AsyncMock()
    index._initialized = True
    return index


@pytest.fixture
def mock_qdrant_search_results():
    """Mock Qdrant search results with proper payload structure."""
    return [
        {
            "id": "uuid-123",
            "score": 0.95,
            "payload": {
                "venue": "polymarket",
                "venue_market_id": "poly-001",
                "identity_hash": "hash-abc",
                "content_hash": "content-xyz",
                "canonical_text": "Market Statement:\nBitcoin to $100k by 2025?\nResolution Criteria:\nThis market resolves to Yes if Bitcoin reaches $100k.",
                "embedding_model": "Qwen/Qwen3-Embedding-4B",
                "embedding_dim": 2048,
            }
        },
        {
            "id": "uuid-456",
            "score": 0.85,
            "payload": {
                "venue": "kalshi",
                "venue_market_id": "kalshi-002",
                "identity_hash": "hash-def",
                "content_hash": "content-uvw",
                "canonical_text": "Market Statement:\nS&P 500 above 6000\nResolution Criteria:\nThis market resolves based on S&P 500 closing price.",
                "embedding_model": "Qwen/Qwen3-Embedding-4B",
                "embedding_dim": 2048,
            }
        }
    ]


@pytest.fixture
def mock_qdrant_search_results_malformed():
    """Mock results with missing/invalid fields for error testing."""
    return [
        {"id": "uuid-1", "score": 0.9, "payload": {}},
        {"id": "uuid-2", "score": 0.8, "payload": {"venue": "invalid"}},
        {"id": "uuid-3", "score": 0.7, "payload": {"venue": "kalshi"}},
    ]


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

