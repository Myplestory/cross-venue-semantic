"""
Pytest configuration and fixtures for orchestrator unit tests.

Every pipeline component is mocked.  Tests verify orchestration logic
(wiring, concurrency, metrics, shutdown, error handling) not component
behaviour.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Path bootstrapping ──────────────────────────────────────────────────
_PIPELINE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from canonicalization.contract_spec import ContractSpec, DateSpec, EntitySpec, ThresholdSpec
from canonicalization.types import CanonicalEvent
from discovery.types import EventType, MarketEvent, OutcomeSpec, VenueType
from embedding.types import EmbeddedEvent
from matching.types import CandidateMatch, VerifiedMatch, VerifiedPair
from persistence.writer import PairWriteRequest

UTC = timezone.utc


# ═══════════════════════════════════════════════════════════════════════
#  Event loop configuration
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    return asyncio.DefaultEventLoopPolicy()


# ═══════════════════════════════════════════════════════════════════════
#  Factory helpers
# ═══════════════════════════════════════════════════════════════════════


def make_market_event(
    venue: VenueType = VenueType.KALSHI,
    market_id: str = "KXBTC-100K-2025",
    title: str = "Will Bitcoin reach $100k by 2025?",
) -> MarketEvent:
    """Create a MarketEvent with sensible defaults."""
    return MarketEvent(
        venue=venue,
        venue_market_id=market_id,
        event_type=EventType.CREATED,
        title=title,
        description="Test market",
        outcomes=[
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No"),
        ],
        received_at=datetime.now(UTC),
    )


def make_canonical_event(
    event: Optional[MarketEvent] = None,
    canonical_text: str = "Market Statement:\nWill Bitcoin reach $100k by 2025?",
    content_hash: str = "hash_content_001",
    identity_hash: str = "hash_identity_001",
) -> CanonicalEvent:
    """Create a CanonicalEvent wrapping a MarketEvent."""
    if event is None:
        event = make_market_event()
    return CanonicalEvent(
        event=event,
        canonical_text=canonical_text,
        content_hash=content_hash,
        identity_hash=identity_hash,
    )


def make_embedded_event(
    canonical_event: Optional[CanonicalEvent] = None,
    embedding_dim: int = 16,
) -> EmbeddedEvent:
    """Create an EmbeddedEvent with a tiny fake embedding."""
    if canonical_event is None:
        canonical_event = make_canonical_event()
    return EmbeddedEvent(
        canonical_event=canonical_event,
        embedding=[0.1] * embedding_dim,
        embedding_model="test-model",
        embedding_dim=embedding_dim,
    )


def make_candidate_match(
    canonical_event: Optional[CanonicalEvent] = None,
    similarity_score: float = 0.9,
) -> CandidateMatch:
    """Create a CandidateMatch from retrieval phase."""
    if canonical_event is None:
        canonical_event = make_canonical_event(
            event=make_market_event(
                venue=VenueType.POLYMARKET,
                market_id="poly-btc-100k-2025",
                title="Bitcoin to $100,000 by end of 2025",
            ),
            content_hash="hash_content_002",
            identity_hash="hash_identity_002",
        )
    return CandidateMatch(
        canonical_event=canonical_event,
        similarity_score=similarity_score,
        embedding=[0.2] * 16,
    )


def make_verified_match(
    candidate: Optional[CandidateMatch] = None,
    cross_encoder_score: float = 0.85,
) -> VerifiedMatch:
    """Create a VerifiedMatch from reranking phase."""
    if candidate is None:
        candidate = make_candidate_match()
    return VerifiedMatch(
        candidate_match=candidate,
        cross_encoder_score=cross_encoder_score,
        match_type="full_match",
        nli_scores={"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05},
        primary_event_score=0.85,
        secondary_clause_score=0.80,
    )


def make_contract_spec(
    statement: str = "Will Bitcoin reach $100k by 2025?",
    confidence: float = 0.95,
) -> ContractSpec:
    """Create a ContractSpec for verification."""
    return ContractSpec(
        statement=statement,
        resolution_date=DateSpec(date=datetime(2025, 12, 31), is_deadline=True),
        entities=[
            EntitySpec(name="Bitcoin", entity_type="other", aliases=["BTC"]),
        ],
        thresholds=[
            ThresholdSpec(value=100000.0, unit="dollars", comparison=">="),
        ],
        resolution_criteria="Resolves Yes if BTC >= $100k",
        data_source="Coinbase",
        outcome_labels=["Yes", "No"],
        confidence=confidence,
    )


def make_verified_pair(
    verdict: str = "equivalent",
    confidence: float = 0.93,
) -> VerifiedPair:
    """Create a VerifiedPair for persistence."""
    return VerifiedPair(
        pair_key="kalshi:KXBTC-100K-2025::polymarket:poly-btc-100k-2025",
        market_a_id="hash_identity_001",
        market_b_id="hash_identity_002",
        contract_spec_a=make_contract_spec(),
        contract_spec_b=make_contract_spec(
            statement="Bitcoin to $100,000 by end of 2025",
            confidence=0.92,
        ),
        outcome_mapping={"YES_A": "YES_B", "NO_A": "NO_B"},
        verdict=verdict,
        confidence=confidence,
        comparison_details={
            "entity_score": 0.95,
            "threshold_score": 1.0,
            "date_score": 1.0,
            "weighted_score": 0.93,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
#  Mock component fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_connector():
    """
    Mock BaseVenueConnector that yields configurable MarketEvents.

    One-shot: yields 3 events on the first call to ``stream_events()``.
    Subsequent calls yield nothing (simulating a reconnection to a
    WebSocket with no new data).  Tests must cooperatively shut down
    the orchestrator to exit the reconnection loop.
    """
    connector = AsyncMock()
    connector.venue_name = VenueType.KALSHI
    connector._running = True
    connector.connect = AsyncMock()
    connector.start = AsyncMock()
    connector.disconnect = AsyncMock()

    events = [
        make_market_event(market_id=f"KXTEST-{i}") for i in range(3)
    ]
    _exhausted = False

    async def _stream():
        nonlocal _exhausted
        if not _exhausted:
            _exhausted = True
            for ev in events:
                yield ev
        # Second+ call: empty stream (reconnected but no new data)

    connector.stream_events = _stream
    return connector


@pytest.fixture
def mock_connector_poly():
    """Mock connector for Polymarket venue (one-shot, 2 events)."""
    connector = AsyncMock()
    connector.venue_name = VenueType.POLYMARKET
    connector._running = True
    connector.connect = AsyncMock()
    connector.start = AsyncMock()
    connector.disconnect = AsyncMock()

    events = [
        make_market_event(
            venue=VenueType.POLYMARKET,
            market_id=f"POLY-{i}",
            title=f"Polymarket event {i}",
        )
        for i in range(2)
    ]
    _exhausted = False

    async def _stream():
        nonlocal _exhausted
        if not _exhausted:
            _exhausted = True
            for ev in events:
                yield ev

    connector.stream_events = _stream
    return connector


@pytest.fixture
def mock_deduplicator():
    """Mock MarketDeduplicator that never marks duplicates."""
    dedup = MagicMock()
    dedup.is_duplicate = MagicMock(return_value=False)
    dedup.size = MagicMock(return_value=0)
    return dedup


@pytest.fixture
def mock_text_builder():
    """Mock CanonicalTextBuilder."""
    builder = MagicMock()
    builder.build = MagicMock(
        return_value="Market Statement:\nTest canonical text"
    )
    return builder


@pytest.fixture
def mock_embedding_encoder():
    """Mock EmbeddingEncoder."""
    encoder = AsyncMock()
    encoder.device = "cpu"
    encoder.initialize = AsyncMock()
    return encoder


@pytest.fixture
def mock_qdrant_index():
    """Mock QdrantIndex."""
    index = AsyncMock()
    index.initialize = AsyncMock()
    return index


@pytest.fixture
def mock_embedding_cache():
    """Mock InMemoryCache for embeddings."""
    cache = AsyncMock()
    cache.initialize = AsyncMock()
    return cache


@pytest.fixture
def mock_embedding_processor():
    """
    Mock EmbeddingProcessor.

    Returns an EmbeddedEvent for any input CanonicalEvent.
    """
    processor = AsyncMock()

    async def _process_async(canonical_event):
        return make_embedded_event(canonical_event=canonical_event)

    processor.process_async = AsyncMock(side_effect=_process_async)
    return processor


@pytest.fixture
def mock_retriever():
    """
    Mock CandidateRetriever.

    Returns 2 candidate matches by default.
    """
    retriever = AsyncMock()

    async def _retrieve(embedded_event, exclude_venue=None):
        return [make_candidate_match(), make_candidate_match()]

    retriever.retrieve_candidates = AsyncMock(side_effect=_retrieve)
    return retriever


@pytest.fixture
def mock_retriever_empty():
    """Mock CandidateRetriever that returns no candidates."""
    retriever = AsyncMock()
    retriever.retrieve_candidates = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder."""
    encoder = AsyncMock()
    encoder.device = "cpu"
    encoder.initialize = AsyncMock()
    return encoder


@pytest.fixture
def mock_reranker():
    """
    Mock CandidateReranker.

    Returns 1 VerifiedMatch per call by default.
    """
    reranker = AsyncMock()

    async def _rerank(query_event, candidates):
        return [make_verified_match()]

    reranker.rerank_async = AsyncMock(side_effect=_rerank)
    return reranker


@pytest.fixture
def mock_reranker_empty():
    """Mock CandidateReranker that returns no matches."""
    reranker = AsyncMock()
    reranker.rerank_async = AsyncMock(return_value=[])
    return reranker


@pytest.fixture
def mock_spec_extractor():
    """
    Mock ContractSpecExtractor.

    Returns a ContractSpec for any input.
    """
    extractor = AsyncMock()
    extractor.initialize = AsyncMock()

    async def _extract(canonical_text, content_hash=None):
        return make_contract_spec(statement=canonical_text[:60])

    extractor.extract_async = AsyncMock(side_effect=_extract)
    return extractor


@pytest.fixture
def mock_pair_verifier():
    """
    Mock PairVerifier.

    Returns 'equivalent' verdict by default.
    """
    verifier = AsyncMock()
    verifier.initialize = AsyncMock()

    async def _verify(
        verified_match, contract_spec_a, contract_spec_b,
        market_a_id, market_b_id,
    ):
        return make_verified_pair(verdict="equivalent")

    verifier.verify_pair_async = AsyncMock(side_effect=_verify)
    return verifier


@pytest.fixture
def mock_pair_verifier_not_equivalent():
    """Mock PairVerifier that returns 'not_equivalent' verdict."""
    verifier = AsyncMock()
    verifier.initialize = AsyncMock()

    async def _verify(
        verified_match, contract_spec_a, contract_spec_b,
        market_a_id, market_b_id,
    ):
        return make_verified_pair(verdict="not_equivalent", confidence=0.3)

    verifier.verify_pair_async = AsyncMock(side_effect=_verify)
    return verifier


@pytest.fixture
def mock_writer():
    """
    Mock PipelineWriter.

    Records enqueue calls for assertion.
    """
    writer = AsyncMock()
    writer.initialize = AsyncMock()
    writer.start = AsyncMock()
    writer.stop = AsyncMock()
    writer.enqueue = AsyncMock()
    writer.get_stats = MagicMock(return_value={
        "batches_written": 0,
        "pairs_written": 0,
        "errors": 0,
        "dlq_size": 0,
        "queue_depth": 0,
        "running": True,
    })
    return writer


@pytest.fixture
def mock_writer_full():
    """Mock PipelineWriter whose enqueue raises QueueFull."""
    writer = AsyncMock()
    writer.initialize = AsyncMock()
    writer.start = AsyncMock()
    writer.stop = AsyncMock()
    writer.enqueue = AsyncMock(side_effect=asyncio.QueueFull())
    writer.get_stats = MagicMock(return_value={
        "batches_written": 0,
        "pairs_written": 0,
        "errors": 0,
        "dlq_size": 0,
        "queue_depth": 100,
        "running": True,
    })
    return writer


# ═══════════════════════════════════════════════════════════════════════
#  Orchestrator fixture (fully wired with mocks)
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def wired_orchestrator(
    mock_connector,
    mock_deduplicator,
    mock_text_builder,
    mock_embedding_encoder,
    mock_qdrant_index,
    mock_embedding_cache,
    mock_embedding_processor,
    mock_retriever,
    mock_cross_encoder,
    mock_reranker,
    mock_spec_extractor,
    mock_pair_verifier,
    mock_writer,
):
    """
    SemanticPipelineOrchestrator with all components pre-injected.

    Skips ``initialize()`` — tests start from a ready state.
    """
    from orchestrator import SemanticPipelineOrchestrator

    orch = SemanticPipelineOrchestrator(
        venues=[VenueType.KALSHI],
        ingestion_queue_size=50,
        num_workers=1,
        model_id="test-model",
        prompt_version="v0.0-test",
    )

    # Inject mocks
    orch._connectors = [mock_connector]
    orch._deduplicator = mock_deduplicator
    orch._text_builders = {VenueType.KALSHI: mock_text_builder}
    orch._embedding_encoder = mock_embedding_encoder
    orch._qdrant_index = mock_qdrant_index
    orch._embedding_cache = mock_embedding_cache
    orch._embedding_processor = mock_embedding_processor
    orch._retriever = mock_retriever
    orch._cross_encoder = mock_cross_encoder
    orch._reranker = mock_reranker
    orch._spec_extractor = mock_spec_extractor
    orch._pair_verifier = mock_pair_verifier
    orch._writer = mock_writer

    return orch

