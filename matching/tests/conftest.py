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


# Cross-Encoder Test Fixtures

@pytest.fixture
def sample_nli_scores():
    """Sample NLI score dictionaries for testing."""
    return {
        "high_entailment": {
            "entailment": 0.9,
            "neutral": 0.05,
            "contradiction": 0.05
        },
        "neutral": {
            "entailment": 0.3,
            "neutral": 0.6,
            "contradiction": 0.1
        },
        "contradiction": {
            "entailment": 0.1,
            "neutral": 0.2,
            "contradiction": 0.7
        },
        "moderate_entailment": {
            "entailment": 0.5,
            "neutral": 0.3,
            "contradiction": 0.2
        }
    }


@pytest.fixture
def mock_nli_pipeline():
    """Mock transformers pipeline that returns NLI scores.
    
    With return_all_scores=True, transformers pipeline returns a flat list of dicts
    for single input: [{label: "...", score: ...}, ...]
    """
    pipeline = MagicMock()
    
    # Default single result (high entailment) - flat list, not nested
    pipeline.return_value = [
        {"label": "ENTAILMENT", "score": 0.9},
        {"label": "NEUTRAL", "score": 0.05},
        {"label": "CONTRADICTION", "score": 0.05}
    ]
    
    return pipeline


@pytest.fixture
def mock_cross_encoder_model():
    """Mock transformers model and tokenizer."""
    model = MagicMock()
    tokenizer = MagicMock()
    
    # Mock model config
    model.config = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    
    return model, tokenizer


@pytest.fixture
def sample_canonical_texts():
    """Sample canonical texts with different structures."""
    return {
        "standard": """Market Statement:
Will Bitcoin reach $100,000 by December 31, 2025?

Resolution Criteria:
This market resolves to Yes if Bitcoin (BTC) reaches or exceeds $100,000 USD on any major exchange (Coinbase, Binance, Kraken) before 11:59 PM UTC on December 31, 2025. The price must be sustained for at least 1 minute. If Bitcoin does not reach this price by the deadline, the market resolves to No.

Clarifications:
- Price data from CoinMarketCap will be used as the reference.
- Any fork or airdrop does not affect the resolution.

End Date: 2025-12-31T23:59:59Z

Outcomes:
- Yes: Bitcoin reaches $100k by end of 2025
- No: Bitcoin does not reach $100k by end of 2025""",
        
        "minimal": """Market Statement:
S&P 500 above 6000

Resolution Criteria:
Based on closing price.""",
        
        "no_clauses": """Market Statement:
Will it rain tomorrow?

End Date: 2025-01-01T00:00:00Z""",
        
        "long_text": """Market Statement:
Will the total market capitalization of all cryptocurrencies combined exceed $5 trillion USD by the end of 2026, measured at 11:59 PM UTC on December 31, 2026, using data from CoinMarketCap as the authoritative source, with the calculation based on the sum of all individual cryptocurrency market capitalizations listed on CoinMarketCap at that specific timestamp, excluding any stablecoins pegged to fiat currencies, and requiring that this threshold be met or exceeded for at least 60 consecutive minutes leading up to the deadline, with verification through multiple independent data sources including but not limited to CoinGecko, Nomics, and CryptoCompare, and accounting for any potential market manipulation or flash crashes that may temporarily spike or depress the total market cap, ensuring that the final determination reflects the true sustained market value rather than transient anomalies, while also considering any potential regulatory changes or market structure modifications that could impact the calculation methodology, and taking into account the potential for new cryptocurrency projects to be launched or existing ones to be delisted during the measurement period, with all such changes being factored into the final calculation using the methodology established at the time of market creation, and with any disputes regarding the final value being resolved through a predetermined arbitration process involving independent financial data providers and market analysts with expertise in cryptocurrency market analysis and valuation methodologies?

Resolution Criteria:
This market resolves based on the total market capitalization of all cryptocurrencies as reported by CoinMarketCap at the specified time, with verification from additional sources.""",
    }


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder instance for unit tests."""
    from matching.cross_encoder import CrossEncoder
    from unittest.mock import MagicMock, AsyncMock
    
    encoder = CrossEncoder()
    encoder.initialize = AsyncMock()
    encoder.score_equivalence_async = AsyncMock(return_value={
        "entailment": 0.9,
        "neutral": 0.05,
        "contradiction": 0.05
    })
    encoder.score_batch_async = AsyncMock(return_value=[{
        "entailment": 0.9,
        "neutral": 0.05,
        "contradiction": 0.05
    }])
    encoder.score_secondary_clauses_async = AsyncMock(return_value=0.8)
    encoder.extract_primary_event = MagicMock(return_value="Extracted primary event")
    encoder.extract_secondary_clauses = MagicMock(return_value=["Clause 1", "Clause 2"])
    encoder.map_nli_to_confidence = MagicMock(return_value=(0.9, "full_match"))
    encoder._initialized = True
    encoder.entailment_threshold = 0.7
    encoder.neutral_threshold = 0.3
    
    return encoder


@pytest.fixture
def sample_candidate_matches(sample_canonical_event, mock_embedding):
    """Sample CandidateMatch objects for reranking tests."""
    from matching.types import CandidateMatch
    from discovery.types import VenueType
    from datetime import datetime, UTC
    
    # Create multiple candidate matches with different similarity scores
    candidates = []
    
    for i in range(5):
        # Create a new canonical event for each candidate
        from canonicalization.types import CanonicalEvent
        from discovery.types import MarketEvent, EventType
        
        market_event = MarketEvent(
            venue=VenueType.POLYMARKET if i % 2 == 0 else VenueType.KALSHI,
            venue_market_id=f"test-market-{i}",
            event_type=EventType.CREATED,
            title=f"Test Market {i}",
            description=f"Description for market {i}",
            received_at=datetime.now(UTC)
        )
        
        canonical = CanonicalEvent(
            event=market_event,
            canonical_text=f"Market Statement:\nTest Market {i}\nResolution Criteria:\nTest resolution.",
            content_hash=f"content-{i}",
            identity_hash=f"identity-{i}"
        )
        
        candidate = CandidateMatch(
            canonical_event=canonical,
            similarity_score=0.95 - (i * 0.1),  # Decreasing scores: 0.95, 0.85, 0.75, 0.65, 0.55
            embedding=mock_embedding,
            retrieval_metadata={"qdrant_id": f"uuid-{i}"}
        )
        
        candidates.append(candidate)
    
    return candidates

