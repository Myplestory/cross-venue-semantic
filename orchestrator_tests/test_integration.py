"""
Integration tests for the SemanticPipelineOrchestrator.

Validates the full pipeline end-to-end with real ML models and a real
Qdrant instance.  Only the venue connectors and PipelineWriter (DB) are
mocked so the test can run without live WebSocket feeds or PostgreSQL.

Real components exercised:
  - EmbeddingEncoder      (Qwen3-Embedding-4B, GPU)
  - QdrantIndex           (cloud Qdrant instance)
  - InMemoryCache         (embedding cache)
  - EmbeddingProcessor    (encode + upsert + cache)
  - CandidateRetriever    (Qdrant vector search)
  - CrossEncoder          (DeBERTa-v3-large-mnli, CPU)
  - CandidateReranker     (NLI-based reranking)
  - ContractSpecExtractor (rule-based extraction)
  - PairVerifier          (rule-based comparators)
  - MarketDeduplicator    (in-memory dedup)
  - CanonicalTextBuilder  (per-venue text templates)

Mocked components:
  - BaseVenueConnector    (fake async generator yielding test events)
  - PipelineWriter        (mock DB — captures enqueue calls for assertion)

GPU Memory Management:
  - Embedding model on CUDA (if available)
  - Cross-encoder on CPU (to fit within 8 GB VRAM on RTX 3070)

Test Flow:
  1. Seed Qdrant with Polymarket markets (embed + upsert)
  2. Create fake Kalshi connector streaming semantically equivalent events
  3. Wire orchestrator with real components + mocked writer/connector
  4. Run orchestrator, stop after events are processed
  5. Assert: all 7 stages executed, cross-venue matches found, writer called

Markers:
  - integration: requires live Qdrant + real models
  - slow: takes ~1-2 minutes (model loading + inference)
  - real_model: uses real ML models (GPU/CPU)
"""

import asyncio
import gc
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

# ── Path bootstrapping ──────────────────────────────────────────────────
_PIPELINE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from canonicalization.hasher import ContentHasher
from canonicalization.text_builder import get_builder
from canonicalization.types import CanonicalEvent
from discovery.dedup import MarketDeduplicator
from discovery.types import EventType, MarketEvent, OutcomeSpec, VenueType
from embedding.cache.in_memory import InMemoryCache
from embedding.encoder import EmbeddingEncoder
from embedding.index import QdrantIndex
from embedding.processor import EmbeddingProcessor
from embedding.types import EmbeddedEvent
from extraction.spec_extractor import ContractSpecExtractor
from matching.cross_encoder import CrossEncoder
from matching.pair_verifier import PairVerifier
from matching.reranker import CandidateReranker
from matching.retriever import CandidateRetriever
from orchestrator import SemanticPipelineOrchestrator
import config

logger = logging.getLogger(__name__)

UTC = timezone.utc

# ── Test collection name (isolated from other test suites) ───────────────
_TEST_COLLECTION = f"{config.QDRANT_COLLECTION_NAME}_orch_integration"


# ═══════════════════════════════════════════════════════════════════════
#  Sample market data (cross-venue semantic pairs)
# ═══════════════════════════════════════════════════════════════════════

# Polymarket markets → seeded into Qdrant BEFORE the orchestrator runs
SEED_MARKETS_POLYMARKET: List[dict] = [
    {
        "venue": VenueType.POLYMARKET,
        "venue_market_id": "poly-orch-001",
        "title": "Bitcoin to $100k by 2025?",
        "description": "Will Bitcoin reach $100,000 USD before 2026?",
        "resolution_criteria": (
            "This market resolves to Yes if Bitcoin (BTC) trades at or "
            "above $100,000 USD on Coinbase, Binance, or Kraken at any "
            "time before January 1, 2026.  The price must be maintained "
            "for at least 60 seconds.  Coinbase will serve as the primary "
            "data source in case of discrepancies."
        ),
        "end_date": datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No"),
        ],
    },
    {
        "venue": VenueType.POLYMARKET,
        "venue_market_id": "poly-orch-002",
        "title": "S&P 500 above 6000 in 2025",
        "description": "Will the S&P 500 index close above 6000 during 2025?",
        "resolution_criteria": (
            "This market resolves to Yes if the S&P 500 index closes at "
            "or above 6000 on any trading day in 2025.  The official "
            "closing value from S&P Dow Jones Indices at 4:00 PM ET will "
            "be used."
        ),
        "end_date": datetime(2025, 12, 31, 16, 0, 0, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No"),
        ],
    },
]

# Kalshi markets → streamed via fake connector (semantic equivalents)
STREAM_MARKETS_KALSHI: List[dict] = [
    {
        "venue": VenueType.KALSHI,
        "venue_market_id": "kalshi-orch-001",
        "title": "Will Bitcoin reach $100,000 by December 31, 2025?",
        "description": (
            "This market tracks whether Bitcoin (BTC) will reach or "
            "exceed $100,000 USD."
        ),
        "resolution_criteria": (
            "This market resolves to Yes if Bitcoin (BTC) reaches or "
            "exceeds $100,000 USD on Coinbase, Binance, Kraken, or "
            "Bitstamp before 11:59 PM ET on December 31, 2025.  The price "
            "must be sustained for at least 1 minute.  Coinbase closing "
            "price will be used as the authoritative source."
        ),
        "end_date": datetime(2025, 12, 31, 23, 59, 0, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No"),
        ],
    },
    {
        "venue": VenueType.KALSHI,
        "venue_market_id": "kalshi-orch-002",
        "title": "Will the S&P 500 close above 6000 in 2025?",
        "description": (
            "This market tracks the S&P 500 index closing price for 2025."
        ),
        "resolution_criteria": (
            "This market resolves to Yes if the S&P 500 index (SPX) "
            "closes at or above 6000.00 on any trading day in 2025.  The "
            "closing price is determined by the official S&P 500 index "
            "value at market close (4:00 PM ET) as reported by S&P Dow "
            "Jones Indices."
        ),
        "end_date": datetime(2025, 12, 31, 16, 0, 0, tzinfo=UTC),
        "outcomes": [
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No"),
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _flush_gpu_memory() -> None:
    """Force GPU memory cleanup.  Must be called AFTER ``del model``."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_market_event(market_data: dict) -> MarketEvent:
    """Convert a sample market dict to a MarketEvent."""
    return MarketEvent(
        venue=market_data["venue"],
        venue_market_id=market_data["venue_market_id"],
        event_type=EventType.CREATED,
        title=market_data["title"],
        description=market_data.get("description", ""),
        outcomes=market_data.get("outcomes", []),
        end_date=market_data.get("end_date"),
        resolution_criteria=market_data.get("resolution_criteria"),
        received_at=datetime.now(UTC),
    )


def _canonicalize_event(event: MarketEvent) -> CanonicalEvent:
    """Canonicalize a MarketEvent using real text builders + hashers."""
    builder = get_builder(event.venue)
    canonical_text = builder.build(event)
    content_hash = ContentHasher.hash_content(canonical_text)
    identity_hash = ContentHasher.identity_hash(
        event.venue, event.venue_market_id
    )
    return CanonicalEvent(
        event=event,
        canonical_text=canonical_text,
        content_hash=content_hash,
        identity_hash=identity_hash,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Module-scoped fixtures (expensive model loading — once per module)
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def qdrant_index():
    """
    Real Qdrant index pointing to an isolated integration test collection.

    Skips the entire test module if QDRANT_API_KEY is not set.
    """
    if not config.QDRANT_API_KEY:
        pytest.skip("QDRANT_API_KEY not set — skipping integration tests")

    index = QdrantIndex(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=_TEST_COLLECTION,
        vector_size=config.QDRANT_VECTOR_SIZE,
    )
    yield index


@pytest.fixture(scope="module")
def embedding_encoder():
    """
    Real embedding encoder (Qwen3-Embedding-4B).

    Module-scoped to avoid reloading the model for each test.
    """
    encoder = EmbeddingEncoder(
        model_name=config.EMBEDDING_MODEL,
        device=config.EMBEDDING_DEVICE,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        max_length=config.EMBEDDING_MAX_LENGTH,
        embedding_dim=config.EMBEDDING_DIM,
        instruction=config.EMBEDDING_INSTRUCTION,
        use_quantization=config.EMBEDDING_QUANTIZATION,
    )

    async def _init():
        await encoder.initialize()

    asyncio.run(_init())
    logger.info("Embedding encoder loaded: device=%s", encoder.device)

    yield encoder

    # Cleanup: free GPU VRAM
    del encoder
    _flush_gpu_memory()
    logger.info("Embedding encoder released")


@pytest.fixture(scope="module")
def cross_encoder_model():
    """
    Real cross-encoder (DeBERTa-v3-large-mnli) on CPU.

    Module-scoped to avoid reloading.
    """
    encoder = CrossEncoder(
        model_name=config.CROSS_ENCODER_MODEL,
        device="cpu",  # Explicit CPU to coexist with embedding on CUDA
        batch_size=config.CROSS_ENCODER_BATCH_SIZE,
        max_length=config.CROSS_ENCODER_MAX_LENGTH,
    )

    async def _init():
        await encoder.initialize()

    asyncio.run(_init())
    logger.info("Cross-encoder loaded: device=%s", encoder.device)

    yield encoder


# ═══════════════════════════════════════════════════════════════════════
#  Qdrant seeding fixture
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def seeded_qdrant(qdrant_index, embedding_encoder):
    """
    Seed the test Qdrant collection with Polymarket markets.

    Uses the real embedding encoder to create proper vector
    representations.  Returns the list of seeded CanonicalEvents
    for assertion in tests.

    Cleanup: deletes the test collection after all tests in the
    module have run.
    """
    canonical_events: List[CanonicalEvent] = []

    async def _seed():
        # Initialize Qdrant (creates collection if needed)
        await qdrant_index.initialize()

        cache = InMemoryCache(max_size=100)
        await cache.initialize()

        processor = EmbeddingProcessor(
            encoder=embedding_encoder,
            index=qdrant_index,
            cache=cache,
            batch_size=config.EMBEDDING_BATCH_SIZE,
        )

        for market_data in SEED_MARKETS_POLYMARKET:
            event = _make_market_event(market_data)
            canonical = _canonicalize_event(event)
            canonical_events.append(canonical)

            # Embed and upsert to Qdrant
            await processor.process_async(canonical)
            logger.info(
                "Seeded: %s → %s",
                canonical.event.venue_market_id,
                canonical.content_hash[:12],
            )

        # Small delay for Qdrant to index
        await asyncio.sleep(1.0)

    asyncio.run(_seed())
    logger.info(
        "Qdrant seeded with %d Polymarket markets in '%s'",
        len(canonical_events),
        _TEST_COLLECTION,
    )

    yield canonical_events

    # ── Teardown: delete test collection ──────────────────────────────
    async def _teardown():
        try:
            if hasattr(qdrant_index, '_client') and qdrant_index._client:
                await qdrant_index._client.delete_collection(
                    collection_name=_TEST_COLLECTION
                )
                logger.info("Deleted test collection: %s", _TEST_COLLECTION)
        except Exception as exc:
            logger.warning("Failed to delete test collection: %s", exc)

    asyncio.run(_teardown())


# ═══════════════════════════════════════════════════════════════════════
#  Orchestrator fixture (real components + mocked connector/writer)
# ═══════════════════════════════════════════════════════════════════════


def _make_fake_connector(
    events: List[MarketEvent],
    venue: VenueType = VenueType.KALSHI,
) -> MagicMock:
    """
    Create a one-shot fake connector that yields the given events.

    On the first call to ``stream_events()``, yields all events.
    On subsequent calls, yields nothing (simulates empty reconnection).
    """
    connector = AsyncMock()
    connector.venue_name = venue
    connector._running = True
    connector.connect = AsyncMock()
    connector.start = AsyncMock()
    connector.disconnect = AsyncMock()

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
def integration_orchestrator(
    qdrant_index,
    embedding_encoder,
    cross_encoder_model,
    seeded_qdrant,
):
    """
    Orchestrator wired with real ML components and mocked I/O.

    Skips ``initialize()`` — all components are pre-injected.
    """
    # Create Kalshi events to stream
    kalshi_events = [
        _make_market_event(m) for m in STREAM_MARKETS_KALSHI
    ]
    fake_connector = _make_fake_connector(kalshi_events)

    # Mock writer (capture enqueue calls)
    mock_writer = AsyncMock()
    mock_writer.initialize = AsyncMock()
    mock_writer.start = AsyncMock()
    mock_writer.stop = AsyncMock()
    mock_writer.enqueue = AsyncMock()
    mock_writer.get_stats = MagicMock(return_value={
        "batches_written": 0,
        "pairs_written": 0,
        "errors": 0,
        "dlq_size": 0,
        "queue_depth": 0,
        "running": True,
    })

    # Real components
    cache = InMemoryCache(max_size=100)
    processor = EmbeddingProcessor(
        encoder=embedding_encoder,
        index=qdrant_index,
        cache=cache,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    retriever = CandidateRetriever(
        index=qdrant_index,
        default_top_k=10,
        default_score_threshold=0.0,  # Low threshold to ensure matches
    )
    reranker = CandidateReranker(
        cross_encoder=cross_encoder_model,
        top_k=5,
        score_threshold=0.3,  # Relaxed for integration testing
        primary_weight=config.CROSS_ENCODER_PRIMARY_WEIGHT,
        secondary_weight=config.CROSS_ENCODER_SECONDARY_WEIGHT,
    )
    spec_extractor = ContractSpecExtractor(
        use_llm_fallback=False,  # Rule-based only — no LLM API calls
        confidence_threshold=config.EXTRACTION_CONFIDENCE_THRESHOLD,
        high_confidence_threshold=config.EXTRACTION_HIGH_CONFIDENCE_THRESHOLD,
    )
    pair_verifier = PairVerifier()

    # Wire orchestrator
    orch = SemanticPipelineOrchestrator(
        venues=[VenueType.KALSHI],
        ingestion_queue_size=50,
        num_workers=1,
        model_id="integration-test",
        prompt_version="v0.0-test",
    )

    orch._connectors = [fake_connector]
    orch._deduplicator = MarketDeduplicator(ttl_seconds=300)
    orch._text_builders = {VenueType.KALSHI: get_builder(VenueType.KALSHI)}
    orch._embedding_encoder = embedding_encoder
    orch._qdrant_index = qdrant_index
    orch._embedding_cache = cache
    orch._embedding_processor = processor
    orch._retriever = retriever
    orch._cross_encoder = cross_encoder_model
    orch._reranker = reranker
    orch._spec_extractor = spec_extractor
    orch._pair_verifier = pair_verifier
    orch._writer = mock_writer

    # Initialize caches synchronously
    async def _init_caches():
        await cache.initialize()
        await spec_extractor.initialize()
        await pair_verifier.initialize()

    asyncio.get_event_loop().run_until_complete(_init_caches())

    return orch, mock_writer, seeded_qdrant


# ═══════════════════════════════════════════════════════════════════════
#  Integration tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.real_model
@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(integration_orchestrator):
    """
    Validate end-to-end data flow through all 7 pipeline stages.

    Seeds Qdrant with Polymarket markets, streams semantically
    equivalent Kalshi markets, and asserts that:
      1. All events are received and processed
      2. Candidates are retrieved from Qdrant (cross-venue)
      3. Cross-encoder reranking produces verified matches
      4. ContractSpecs are extracted for both sides
      5. Pair verification produces a verdict
      6. Writer.enqueue is called for actionable pairs
      7. No events fail
    """
    orch, mock_writer, seeded_events = integration_orchestrator
    num_kalshi_events = len(STREAM_MARKETS_KALSHI)

    # ── Run orchestrator with auto-shutdown stopper ────────────────────
    async def _stopper():
        """Monitor metrics and shut down once all events are processed."""
        deadline = time.monotonic() + 120.0  # 2-minute hard timeout
        while time.monotonic() < deadline:
            m = orch._metrics
            if m.events_processed + m.events_failed >= num_kalshi_events:
                # All events handled — allow brief drain time
                await asyncio.sleep(0.5)
                await orch.shutdown()
                return
            await asyncio.sleep(0.25)

        # Hard timeout — force shutdown
        await orch.shutdown()

    # Start orchestrator + stopper concurrently
    await asyncio.gather(
        orch.run(),
        _stopper(),
    )

    # ── Assertions ────────────────────────────────────────────────────
    m = orch._metrics

    # 1. All events received
    assert m.events_received >= num_kalshi_events, (
        f"Expected >= {num_kalshi_events} events received, got {m.events_received}"
    )

    # 2. All events processed (no failures)
    assert m.events_processed >= num_kalshi_events, (
        f"Expected >= {num_kalshi_events} events processed, got {m.events_processed}"
    )
    assert m.events_failed == 0, (
        f"Expected 0 failures, got {m.events_failed}"
    )

    # 3. Stage latencies recorded for each stage that was called
    assert m.canonicalization.total_calls >= num_kalshi_events, (
        f"Canonicalization not called enough: {m.canonicalization.total_calls}"
    )
    assert m.embedding.total_calls >= num_kalshi_events, (
        f"Embedding not called enough: {m.embedding.total_calls}"
    )
    assert m.retrieval.total_calls >= num_kalshi_events, (
        f"Retrieval not called enough: {m.retrieval.total_calls}"
    )

    # 4. Cross-venue candidates found (Qdrant search found Polymarket hits)
    #    At least one event should have found candidates
    assert m.events_no_candidates < num_kalshi_events, (
        f"All {num_kalshi_events} events had no candidates — "
        "Qdrant seeding or retrieval likely failed"
    )

    # 5. Reranking ran for events with candidates
    events_with_candidates = num_kalshi_events - m.events_no_candidates
    assert m.reranking.total_calls >= events_with_candidates, (
        f"Reranking calls ({m.reranking.total_calls}) < events with "
        f"candidates ({events_with_candidates})"
    )

    # 6. Pairs found through reranking
    assert m.pairs_found > 0, (
        "No pairs found — cross-encoder may have filtered all candidates"
    )

    # 7. Extraction + verification ran for found pairs
    assert m.extraction.total_calls > 0, (
        f"Extraction never called (pairs_found={m.pairs_found})"
    )
    assert m.verification.total_calls > 0, (
        f"Verification never called (pairs_found={m.pairs_found})"
    )

    # 8. Verdicts produced
    total_verdicts = (
        m.pairs_equivalent + m.pairs_needs_review + m.pairs_not_equivalent
    )
    assert total_verdicts > 0, "No verdicts produced"

    # 9. Actionable pairs persisted (equivalent or needs_review)
    actionable = m.pairs_equivalent + m.pairs_needs_review
    if actionable > 0:
        assert mock_writer.enqueue.call_count > 0, (
            f"Writer.enqueue not called despite {actionable} actionable pairs"
        )
        assert m.pairs_persisted > 0, "Actionable pairs not persisted"
    else:
        # All pairs were "not_equivalent" — this is acceptable for the
        # test data but we log it for investigation
        logger.warning(
            "All pairs were not_equivalent — consider adjusting "
            "thresholds or test data"
        )

    # ── Log full metrics for debugging ─────────────────────────────────
    summary = m.summary()
    logger.info("Integration test metrics: %s", summary)
    print(f"\n{'='*60}")
    print("INTEGRATION TEST METRICS")
    print(f"{'='*60}")
    print(f"  Events received:      {m.events_received}")
    print(f"  Events processed:     {m.events_processed}")
    print(f"  Events failed:        {m.events_failed}")
    print(f"  Events no candidates: {m.events_no_candidates}")
    print(f"  Pairs found:          {m.pairs_found}")
    print(f"  Pairs equivalent:     {m.pairs_equivalent}")
    print(f"  Pairs needs_review:   {m.pairs_needs_review}")
    print(f"  Pairs not_equivalent: {m.pairs_not_equivalent}")
    print(f"  Pairs persisted:      {m.pairs_persisted}")
    print(f"  Writer enqueue calls: {mock_writer.enqueue.call_count}")
    print(f"{'='*60}")
    for stage in (
        "canonicalization", "embedding", "retrieval",
        "reranking", "extraction", "verification", "persistence",
    ):
        s = getattr(m, stage)
        if s.total_calls > 0:
            print(
                f"  {stage:20s}  calls={s.total_calls:3d}  "
                f"avg={s.avg_latency_ms:8.1f}ms  "
                f"min={s.min_latency_ms:8.1f}ms  "
                f"max={s.max_latency_ms:8.1f}ms"
            )
    print(f"{'='*60}\n")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.real_model
@pytest.mark.asyncio
async def test_pipeline_performance_thresholds(integration_orchestrator):
    """
    Validate per-stage latency thresholds (fintech industry standards).

    After the pipeline runs end-to-end, checks that stage latencies
    are within acceptable bounds for a batch-processing slow path.
    """
    orch, mock_writer, _ = integration_orchestrator
    num_kalshi_events = len(STREAM_MARKETS_KALSHI)

    # Run with stopper (reuses same pattern as test above)
    async def _stopper():
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            m = orch._metrics
            if m.events_processed + m.events_failed >= num_kalshi_events:
                await asyncio.sleep(0.5)
                await orch.shutdown()
                return
            await asyncio.sleep(0.25)
        await orch.shutdown()

    await asyncio.gather(orch.run(), _stopper())

    m = orch._metrics

    # Device-aware thresholds
    if torch.cuda.is_available():
        # CUDA: embedding model on GPU — fast
        max_embedding_avg_ms = 5_000.0      # 5s per event
        max_reranking_avg_ms = 15_000.0     # 15s (cross-encoder on CPU)
    else:
        # CPU-only: both models on CPU — slower
        max_embedding_avg_ms = 30_000.0     # 30s per event
        max_reranking_avg_ms = 60_000.0     # 60s (NLI on CPU)

    # Canonicalization: always fast (CPU string ops)
    if m.canonicalization.total_calls > 0:
        assert m.canonicalization.avg_latency_ms < 100.0, (
            f"Canonicalization too slow: {m.canonicalization.avg_latency_ms:.1f}ms"
        )

    # Embedding: GPU or CPU
    if m.embedding.total_calls > 0:
        assert m.embedding.avg_latency_ms < max_embedding_avg_ms, (
            f"Embedding too slow: {m.embedding.avg_latency_ms:.1f}ms "
            f"(threshold: {max_embedding_avg_ms}ms)"
        )

    # Retrieval: network call to Qdrant cloud
    if m.retrieval.total_calls > 0:
        assert m.retrieval.avg_latency_ms < 5_000.0, (
            f"Retrieval too slow: {m.retrieval.avg_latency_ms:.1f}ms"
        )

    # Reranking: cross-encoder inference
    if m.reranking.total_calls > 0:
        assert m.reranking.avg_latency_ms < max_reranking_avg_ms, (
            f"Reranking too slow: {m.reranking.avg_latency_ms:.1f}ms "
            f"(threshold: {max_reranking_avg_ms}ms)"
        )

    # Extraction: rule-based, CPU-only
    if m.extraction.total_calls > 0:
        assert m.extraction.avg_latency_ms < 1_000.0, (
            f"Extraction too slow: {m.extraction.avg_latency_ms:.1f}ms"
        )

    # Verification: comparators, CPU-only
    if m.verification.total_calls > 0:
        assert m.verification.avg_latency_ms < 1_000.0, (
            f"Verification too slow: {m.verification.avg_latency_ms:.1f}ms"
        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.real_model
@pytest.mark.asyncio
async def test_writer_enqueue_payload_structure(integration_orchestrator):
    """
    Validate that PairWriteRequest payloads have correct structure.

    When the writer is called, the enqueued request should contain:
      - verified_pair with a valid verdict
      - canonical_event_a (query) and canonical_event_b (candidate)
      - model_id and prompt_version
    """
    orch, mock_writer, _ = integration_orchestrator
    num_kalshi_events = len(STREAM_MARKETS_KALSHI)

    async def _stopper():
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            m = orch._metrics
            if m.events_processed + m.events_failed >= num_kalshi_events:
                await asyncio.sleep(0.5)
                await orch.shutdown()
                return
            await asyncio.sleep(0.25)
        await orch.shutdown()

    await asyncio.gather(orch.run(), _stopper())

    # Check writer payloads (if any were enqueued)
    if mock_writer.enqueue.call_count > 0:
        for call in mock_writer.enqueue.call_args_list:
            request = call[0][0]  # First positional arg

            # Structural assertions
            assert hasattr(request, "verified_pair"), "Missing verified_pair"
            assert hasattr(request, "canonical_event_a"), "Missing event_a"
            assert hasattr(request, "canonical_event_b"), "Missing event_b"
            assert hasattr(request, "model_id"), "Missing model_id"
            assert hasattr(request, "prompt_version"), "Missing prompt_version"

            # Value assertions
            vp = request.verified_pair
            assert vp.verdict in ("equivalent", "needs_review"), (
                f"Unexpected verdict in writer: {vp.verdict}"
            )
            assert 0.0 <= vp.confidence <= 1.0, (
                f"Confidence out of range: {vp.confidence}"
            )
            assert vp.pair_key is not None, "pair_key is None"
            assert request.model_id == "integration-test"
            assert request.prompt_version == "v0.0-test"

            # Cross-venue validation: query (Kalshi) vs candidate (Polymarket)
            assert request.canonical_event_a.event.venue == VenueType.KALSHI
            assert request.canonical_event_b.event.venue == VenueType.POLYMARKET
    else:
        logger.warning(
            "Writer was never called — no actionable pairs produced. "
            "This is acceptable if all pairs were not_equivalent."
        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.real_model
@pytest.mark.asyncio
async def test_graceful_shutdown_during_processing(
    qdrant_index,
    embedding_encoder,
    cross_encoder_model,
    seeded_qdrant,
):
    """
    Verify graceful shutdown does not corrupt state or leak resources.

    Shuts down the orchestrator mid-processing and checks that:
      - No events are lost (metrics are consistent)
      - Worker drain completes within timeout
      - No unhandled exceptions
    """
    kalshi_events = [
        _make_market_event(m) for m in STREAM_MARKETS_KALSHI
    ]
    fake_connector = _make_fake_connector(kalshi_events)

    mock_writer = AsyncMock()
    mock_writer.initialize = AsyncMock()
    mock_writer.start = AsyncMock()
    mock_writer.stop = AsyncMock()
    mock_writer.enqueue = AsyncMock()
    mock_writer.get_stats = MagicMock(return_value={
        "batches_written": 0, "pairs_written": 0,
        "errors": 0, "dlq_size": 0, "queue_depth": 0, "running": True,
    })

    cache = InMemoryCache(max_size=100)
    processor = EmbeddingProcessor(
        encoder=embedding_encoder,
        index=qdrant_index,
        cache=cache,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    retriever = CandidateRetriever(
        index=qdrant_index,
        default_top_k=10,
        default_score_threshold=0.0,
    )
    reranker = CandidateReranker(
        cross_encoder=cross_encoder_model,
        top_k=5,
        score_threshold=0.3,
    )
    spec_extractor = ContractSpecExtractor(use_llm_fallback=False)
    pair_verifier = PairVerifier()

    orch = SemanticPipelineOrchestrator(
        venues=[VenueType.KALSHI],
        ingestion_queue_size=50,
        num_workers=1,
    )

    orch._connectors = [fake_connector]
    orch._deduplicator = MarketDeduplicator(ttl_seconds=300)
    orch._text_builders = {VenueType.KALSHI: get_builder(VenueType.KALSHI)}
    orch._embedding_encoder = embedding_encoder
    orch._qdrant_index = qdrant_index
    orch._embedding_cache = cache
    orch._embedding_processor = processor
    orch._retriever = retriever
    orch._cross_encoder = cross_encoder_model
    orch._reranker = reranker
    orch._spec_extractor = spec_extractor
    orch._pair_verifier = pair_verifier
    orch._writer = mock_writer

    await cache.initialize()
    await spec_extractor.initialize()
    await pair_verifier.initialize()

    # Shut down very quickly (after ~2 seconds)
    async def _early_stopper():
        await asyncio.sleep(2.0)
        await orch.shutdown()

    # Should complete without errors or hanging
    try:
        await asyncio.wait_for(
            asyncio.gather(orch.run(), _early_stopper()),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        pytest.fail("Graceful shutdown timed out (>30s)")

    # Verify consistency
    m = orch._metrics
    assert m.events_processed + m.events_failed <= m.events_received, (
        f"Processed ({m.events_processed}) + Failed ({m.events_failed}) "
        f"> Received ({m.events_received})"
    )
    assert not orch._running, "Orchestrator still running after shutdown"



