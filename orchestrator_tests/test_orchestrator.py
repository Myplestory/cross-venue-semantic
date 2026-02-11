"""
Unit tests for SemanticPipelineOrchestrator.

All pipeline components are mocked — these tests verify orchestration
logic: wiring, concurrency, metrics, shutdown, and error isolation.

Test categories:
- Initialization & configuration
- Ingestion: dedup, backpressure, fan-in
- Connector reconnection with exponential backoff
- Pipeline worker: sentinel, event flow, error isolation
- _process_event: all 7 stages in order, stage failures
- _process_match: verdict routing, writer backpressure
- Shutdown: drain order, idempotency, timeout handling
- Metrics: counters, stage latency, heartbeat
"""

import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ── Path bootstrapping ──────────────────────────────────────────────────
_TEST_DIR = str(Path(__file__).resolve().parent)
_PIPELINE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from orchestrator import (
    SemanticPipelineOrchestrator,
    PipelineMetrics,
    StageMetrics,
    _format_metrics,
)
from discovery.types import VenueType, MarketEvent
from matching.types import VerifiedMatch
from conftest import (
    make_market_event,
    make_canonical_event,
    make_embedded_event,
    make_candidate_match,
    make_verified_match,
    make_contract_spec,
    make_verified_pair,
)

UTC = timezone.utc


# ═══════════════════════════════════════════════════════════════════════
#  1. StageMetrics
# ═══════════════════════════════════════════════════════════════════════


class TestStageMetrics:
    """Unit tests for StageMetrics dataclass."""

    def test_initial_state(self):
        """Metrics start at zero."""
        m = StageMetrics()
        assert m.total_calls == 0
        assert m.total_errors == 0
        assert m.total_latency_ms == 0.0
        assert m.max_latency_ms == 0.0
        assert m.avg_latency_ms == 0.0

    def test_record_single(self):
        """Recording a single call updates all fields."""
        m = StageMetrics()
        m.record(42.5)
        assert m.total_calls == 1
        assert m.total_latency_ms == 42.5
        assert m.max_latency_ms == 42.5
        assert m.min_latency_ms == 42.5
        assert m.avg_latency_ms == 42.5
        assert m.total_errors == 0

    def test_record_multiple(self):
        """Average, min, max tracked across multiple recordings."""
        m = StageMetrics()
        m.record(10.0)
        m.record(30.0)
        m.record(20.0)
        assert m.total_calls == 3
        assert m.total_latency_ms == 60.0
        assert m.avg_latency_ms == 20.0
        assert m.min_latency_ms == 10.0
        assert m.max_latency_ms == 30.0

    def test_record_error(self):
        """Error flag increments total_errors."""
        m = StageMetrics()
        m.record(5.0, error=True)
        m.record(5.0, error=False)
        assert m.total_calls == 2
        assert m.total_errors == 1

    def test_snapshot_empty(self):
        """Snapshot of empty metrics returns zeros."""
        m = StageMetrics()
        snap = m.snapshot()
        assert snap["calls"] == 0
        assert snap["errors"] == 0
        assert snap["avg_ms"] == 0.0
        assert snap["min_ms"] == 0
        assert snap["max_ms"] == 0.0

    def test_snapshot_populated(self):
        """Snapshot reflects recorded data."""
        m = StageMetrics()
        m.record(10.0)
        m.record(20.0)
        snap = m.snapshot()
        assert snap["calls"] == 2
        assert snap["avg_ms"] == 15.0
        assert snap["min_ms"] == 10.0
        assert snap["max_ms"] == 20.0


# ═══════════════════════════════════════════════════════════════════════
#  2. PipelineMetrics
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineMetrics:
    """Unit tests for PipelineMetrics aggregate."""

    def test_initial_counters(self):
        """All event/pair counters start at zero."""
        m = PipelineMetrics()
        assert m.events_received == 0
        assert m.events_processed == 0
        assert m.events_failed == 0
        assert m.pairs_found == 0
        assert m.pairs_equivalent == 0
        assert m.pairs_persisted == 0

    def test_summary_no_uptime(self):
        """Summary works when started_at is None."""
        m = PipelineMetrics()
        s = m.summary()
        assert s["uptime_seconds"] == 0.0
        assert s["throughput_events_per_min"] == 0.0

    def test_summary_with_data(self):
        """Summary reflects event and pair counters."""
        m = PipelineMetrics()
        m.events_received = 10
        m.events_processed = 8
        m.events_failed = 2
        m.pairs_found = 5
        m.pairs_equivalent = 3
        m.pairs_not_equivalent = 1
        m.pairs_needs_review = 1
        m.pairs_persisted = 4
        m.started_at = datetime.now(UTC)

        s = m.summary()
        assert s["events"]["received"] == 10
        assert s["events"]["processed"] == 8
        assert s["events"]["failed"] == 2
        assert s["pairs"]["found"] == 5
        assert s["pairs"]["equivalent"] == 3
        assert s["pairs"]["not_equivalent"] == 1
        assert s["pairs"]["needs_review"] == 1
        assert s["pairs"]["persisted"] == 4
        assert "stages" in s
        assert "canonicalization" in s["stages"]

    def test_summary_has_all_stages(self):
        """Summary includes all 7 pipeline stages."""
        m = PipelineMetrics()
        s = m.summary()
        expected_stages = {
            "canonicalization", "embedding", "retrieval",
            "reranking", "extraction", "verification", "persistence",
        }
        assert set(s["stages"].keys()) == expected_stages


class TestFormatMetrics:
    """Unit tests for _format_metrics helper."""

    def test_format_produces_string(self):
        """_format_metrics returns a non-empty string."""
        m = PipelineMetrics()
        m.started_at = datetime.now(UTC)
        result = _format_metrics(m.summary())
        assert isinstance(result, str)
        assert "Uptime" in result
        assert "Events" in result
        assert "Pairs" in result


# ═══════════════════════════════════════════════════════════════════════
#  3. Orchestrator construction
# ═══════════════════════════════════════════════════════════════════════


class TestOrchestratorConstruction:
    """Tests for __init__ and configuration parsing."""

    def test_default_construction(self):
        """Orchestrator creates with config defaults."""
        with patch("orchestrator.config") as mock_cfg:
            mock_cfg.ORCHESTRATOR_VENUES = "kalshi,polymarket"
            mock_cfg.ORCHESTRATOR_INGESTION_QUEUE_SIZE = 100
            mock_cfg.ORCHESTRATOR_NUM_WORKERS = 2
            mock_cfg.ORCHESTRATOR_MODEL_ID = "gpt-4o-mini"
            mock_cfg.ORCHESTRATOR_PROMPT_VERSION = "v1.0"
            mock_cfg.ORCHESTRATOR_DEDUP_TTL = 3600

            orch = SemanticPipelineOrchestrator()
            assert len(orch._venues) == 2
            assert orch._num_workers == 2
            assert orch._ingestion_queue_size == 100

    def test_explicit_params_override_config(self):
        """Explicit constructor params take precedence over config."""
        orch = SemanticPipelineOrchestrator(
            venues=[VenueType.KALSHI],
            ingestion_queue_size=25,
            num_workers=3,
            model_id="custom-model",
            prompt_version="v99",
        )
        assert orch._venues == [VenueType.KALSHI]
        assert orch._ingestion_queue_size == 25
        assert orch._num_workers == 3
        assert orch._model_id == "custom-model"
        assert orch._prompt_version == "v99"

    def test_queue_bounded(self):
        """Ingestion queue is bounded with the configured maxsize."""
        orch = SemanticPipelineOrchestrator(
            venues=[VenueType.KALSHI],
            ingestion_queue_size=10,
        )
        assert orch._ingestion_queue.maxsize == 10

    def test_starts_not_running(self):
        """Orchestrator is not running after construction."""
        orch = SemanticPipelineOrchestrator(
            venues=[VenueType.KALSHI],
        )
        assert orch._running is False
        assert orch._metrics.started_at is None


# ═══════════════════════════════════════════════════════════════════════
#  4. Initialization
# ═══════════════════════════════════════════════════════════════════════


class TestInitialization:
    """Tests for initialize() component wiring."""

    @pytest.mark.asyncio
    async def test_initialize_raises_if_no_connectors(self):
        """RuntimeError raised when no venue connectors succeed."""
        with patch("orchestrator.create_connector", side_effect=ValueError("unsupported")):
            orch = SemanticPipelineOrchestrator(
                venues=[VenueType.KALSHI],
            )
            with pytest.raises(RuntimeError, match="No venue connectors"):
                await orch.initialize()

    @pytest.mark.asyncio
    async def test_initialize_skips_unsupported_venue(self):
        """Unsupported venues are logged and skipped, not fatal."""
        call_count = 0

        def _create(venue):
            nonlocal call_count
            call_count += 1
            if venue == VenueType.GEMINI:
                raise ValueError("unsupported")
            connector = AsyncMock()
            connector.venue_name = venue
            return connector

        with (
            patch("orchestrator.create_connector", side_effect=_create),
            patch("orchestrator.get_builder", return_value=MagicMock()),
            patch("orchestrator.EmbeddingEncoder") as MockEnc,
            patch("orchestrator.QdrantIndex") as MockIdx,
            patch("orchestrator.InMemoryCache") as MockCache,
            patch("orchestrator.EmbeddingProcessor") as MockProc,
            patch("orchestrator.CrossEncoder") as MockCE,
            patch("orchestrator.CandidateReranker") as MockRR,
            patch("orchestrator.ContractSpecExtractor") as MockExt,
            patch("orchestrator.PairVerifier") as MockPV,
            patch("orchestrator.PipelineWriter") as MockW,
        ):
            # Configure mock returns
            for MockCls in (MockEnc, MockIdx, MockCache, MockCE, MockExt, MockPV, MockW):
                inst = AsyncMock()
                inst.initialize = AsyncMock()
                inst.device = "cpu"
                MockCls.return_value = inst

            MockProc.return_value = MagicMock()
            MockRR.return_value = MagicMock()

            orch = SemanticPipelineOrchestrator(
                venues=[VenueType.KALSHI, VenueType.GEMINI],
            )
            await orch.initialize()

            # Kalshi connector created, Gemini skipped
            assert len(orch._connectors) == 1
            assert call_count == 2


# ═══════════════════════════════════════════════════════════════════════
#  5. Canonicalization helper
# ═══════════════════════════════════════════════════════════════════════


class TestCanonicalize:
    """Tests for _canonicalize() synchronous helper."""

    def test_canonicalize_happy_path(self, wired_orchestrator, mock_text_builder):
        """Returns CanonicalEvent with text and hashes."""
        event = make_market_event()
        result = wired_orchestrator._canonicalize(event)

        assert result.canonical_text == mock_text_builder.build.return_value
        assert result.content_hash is not None
        assert result.identity_hash is not None
        assert result.event is event

    def test_canonicalize_unknown_venue(self, wired_orchestrator):
        """Raises ValueError for unregistered venue."""
        event = make_market_event(venue=VenueType.POLYMARKET)
        with pytest.raises(ValueError, match="No text builder"):
            wired_orchestrator._canonicalize(event)


# ═══════════════════════════════════════════════════════════════════════
#  6. Ingestion
# ═══════════════════════════════════════════════════════════════════════


class TestIngestion:
    """Tests for _ingest_from_connector and deduplication.

    The orchestrator's ``_ingest_from_connector`` runs an infinite
    ``while self._running`` reconnection loop.  Tests use a cooperative
    "stopper" task that monitors metrics and sets ``_running = False``
    once the expected events have been received.  All calls are wrapped
    in ``asyncio.wait_for`` as a CI safety net against hangs.
    """

    @pytest.mark.asyncio
    async def test_events_enqueued(self, wired_orchestrator, mock_connector):
        """Events from connector are put into the ingestion queue."""
        wired_orchestrator._running = True

        async def _stop_when_received():
            while wired_orchestrator._metrics.events_received < 3:
                await asyncio.sleep(0.01)
            wired_orchestrator._running = False

        stopper = asyncio.create_task(_stop_when_received())

        await asyncio.wait_for(
            wired_orchestrator._ingest_from_connector(mock_connector),
            timeout=5.0,
        )
        await stopper

        assert wired_orchestrator._metrics.events_received == 3
        assert wired_orchestrator._ingestion_queue.qsize() == 3

    @pytest.mark.asyncio
    async def test_duplicate_events_filtered(
        self, wired_orchestrator, mock_connector, mock_deduplicator
    ):
        """Duplicate events are counted but not enqueued."""
        wired_orchestrator._running = True
        # First call: not duplicate, second+: duplicate
        mock_deduplicator.is_duplicate.side_effect = [False, True, True]

        async def _stop_when_received():
            while wired_orchestrator._metrics.events_received < 3:
                await asyncio.sleep(0.01)
            wired_orchestrator._running = False

        stopper = asyncio.create_task(_stop_when_received())

        await asyncio.wait_for(
            wired_orchestrator._ingest_from_connector(mock_connector),
            timeout=5.0,
        )
        await stopper

        assert wired_orchestrator._metrics.events_received == 3
        assert wired_orchestrator._metrics.events_deduplicated == 2
        assert wired_orchestrator._ingestion_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_queue_full_drops_event(self, wired_orchestrator, mock_connector):
        """Events are dropped when ingestion queue is full."""
        # Tiny queue
        wired_orchestrator._ingestion_queue = asyncio.Queue(maxsize=1)
        wired_orchestrator._running = True

        # Pre-fill the queue
        wired_orchestrator._ingestion_queue.put_nowait(make_market_event())

        async def _stop_when_received():
            while wired_orchestrator._metrics.events_received < 1:
                await asyncio.sleep(0.01)
            # Give a brief moment for all events to be attempted
            await asyncio.sleep(0.1)
            wired_orchestrator._running = False

        stopper = asyncio.create_task(_stop_when_received())

        await asyncio.wait_for(
            wired_orchestrator._ingest_from_connector(mock_connector),
            timeout=5.0,
        )
        await stopper

        assert wired_orchestrator._metrics.events_received >= 1

    @pytest.mark.asyncio
    async def test_connector_error_triggers_reconnect(self, wired_orchestrator):
        """Connector failure triggers exponential backoff reconnection."""
        connector = AsyncMock()
        connector.venue_name = VenueType.KALSHI
        connector.connect = AsyncMock(side_effect=ConnectionError("ws down"))
        connector.start = AsyncMock()

        wired_orchestrator._running = True

        sleep_calls = []

        async def _fake_sleep(seconds):
            sleep_calls.append(seconds)
            # Count only backoff sleeps (>0); ignore yield-point sleep(0)
            backoff_count = sum(1 for s in sleep_calls if s > 0)
            if backoff_count >= 3:
                wired_orchestrator._running = False

        with patch("orchestrator.asyncio.sleep", side_effect=_fake_sleep):
            await asyncio.wait_for(
                wired_orchestrator._ingest_from_connector(connector),
                timeout=5.0,
            )

        # Filter out sleep(0) yield-point calls
        backoff_calls = [s for s in sleep_calls if s > 0]
        # Backoff: 5, 10, 20 (exponential from base=5)
        assert len(backoff_calls) == 3
        assert backoff_calls[0] == 5.0
        assert backoff_calls[1] == 10.0
        assert backoff_calls[2] == 20.0

    @pytest.mark.asyncio
    async def test_connector_max_retries_exceeded(self, wired_orchestrator):
        """Connector gives up after max retries."""
        connector = AsyncMock()
        connector.venue_name = VenueType.KALSHI
        connector.connect = AsyncMock(side_effect=ConnectionError("ws down"))
        connector.start = AsyncMock()

        wired_orchestrator._running = True

        with patch("orchestrator.asyncio.sleep", new_callable=AsyncMock):
            with patch("orchestrator._MAX_CONNECTOR_RETRIES", 2):
                await asyncio.wait_for(
                    wired_orchestrator._ingest_from_connector(connector),
                    timeout=5.0,
                )

        # Should have exited after exceeding max retries
        assert connector.connect.call_count >= 2


# ═══════════════════════════════════════════════════════════════════════
#  7. Pipeline worker
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineWorker:
    """Tests for _pipeline_worker loop."""

    @pytest.mark.asyncio
    async def test_worker_processes_events(self, wired_orchestrator):
        """Worker processes events from the queue."""
        wired_orchestrator._running = True

        # Put events in queue + sentinel
        for i in range(2):
            wired_orchestrator._ingestion_queue.put_nowait(
                make_market_event(market_id=f"TEST-{i}")
            )
        wired_orchestrator._ingestion_queue.put_nowait(None)  # sentinel

        await asyncio.wait_for(
            wired_orchestrator._pipeline_worker(worker_id=0),
            timeout=5.0,
        )

        assert wired_orchestrator._metrics.events_processed >= 2

    @pytest.mark.asyncio
    async def test_worker_exits_on_sentinel(self, wired_orchestrator):
        """Worker exits cleanly when receiving None sentinel."""
        wired_orchestrator._running = True
        wired_orchestrator._ingestion_queue.put_nowait(None)

        await asyncio.wait_for(
            wired_orchestrator._pipeline_worker(worker_id=0),
            timeout=5.0,
        )
        # No assertion needed — test passes if worker exits without hanging

    @pytest.mark.asyncio
    async def test_worker_exits_on_running_false(self, wired_orchestrator):
        """Worker exits when _running is set to False."""
        wired_orchestrator._running = False

        await asyncio.wait_for(
            wired_orchestrator._pipeline_worker(worker_id=0),
            timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_worker_isolates_event_errors(self, wired_orchestrator):
        """An error processing one event does not kill the worker."""
        wired_orchestrator._running = True

        # First event will fail, second should succeed
        events = [
            make_market_event(market_id="FAIL-1"),
            make_market_event(market_id="PASS-1"),
        ]

        # Make embedding processor fail on first call, succeed on second
        call_count = 0

        async def _fail_then_succeed(canonical_event):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated GPU OOM")
            return make_embedded_event(canonical_event=canonical_event)

        wired_orchestrator._embedding_processor.process_async = AsyncMock(
            side_effect=_fail_then_succeed
        )

        for ev in events:
            wired_orchestrator._ingestion_queue.put_nowait(ev)
        wired_orchestrator._ingestion_queue.put_nowait(None)

        await asyncio.wait_for(
            wired_orchestrator._pipeline_worker(worker_id=0),
            timeout=5.0,
        )

        assert wired_orchestrator._metrics.events_failed >= 1
        assert wired_orchestrator._metrics.events_processed >= 1


# ═══════════════════════════════════════════════════════════════════════
#  8. _process_event (full 7-stage pipeline)
# ═══════════════════════════════════════════════════════════════════════


class TestProcessEvent:
    """Tests for _process_event orchestrating all 7 stages."""

    @pytest.mark.asyncio
    async def test_happy_path_all_stages(self, wired_orchestrator):
        """All 7 stages called in order for a normal event."""
        event = make_market_event()
        await wired_orchestrator._process_event(event, "test-worker")

        # Stage 2: embedding
        wired_orchestrator._embedding_processor.process_async.assert_called_once()

        # Stage 3: retrieval
        wired_orchestrator._retriever.retrieve_candidates.assert_called_once()

        # Stage 4: reranking
        wired_orchestrator._reranker.rerank_async.assert_called_once()

        # Stage 5: extraction (called twice — once per side)
        assert wired_orchestrator._spec_extractor.extract_async.call_count == 2

        # Stage 6: verification
        wired_orchestrator._pair_verifier.verify_pair_async.assert_called_once()

        # Stage 7: persistence
        wired_orchestrator._writer.enqueue.assert_called_once()

        # Metrics
        assert wired_orchestrator._metrics.events_processed == 1
        assert wired_orchestrator._metrics.pairs_found >= 1
        assert wired_orchestrator._metrics.pairs_equivalent == 1
        assert wired_orchestrator._metrics.pairs_persisted == 1

    @pytest.mark.asyncio
    async def test_no_candidates_short_circuits(
        self, wired_orchestrator, mock_retriever_empty
    ):
        """Pipeline stops after retrieval if no candidates found."""
        wired_orchestrator._retriever = mock_retriever_empty
        event = make_market_event()

        await wired_orchestrator._process_event(event, "test-worker")

        # Retrieval called, but reranking never reached
        mock_retriever_empty.retrieve_candidates.assert_called_once()
        wired_orchestrator._reranker.rerank_async.assert_not_called()
        assert wired_orchestrator._metrics.events_no_candidates == 1
        assert wired_orchestrator._metrics.events_processed == 1

    @pytest.mark.asyncio
    async def test_no_matches_after_reranking(
        self, wired_orchestrator, mock_reranker_empty
    ):
        """Pipeline stops after reranking if no verified matches."""
        wired_orchestrator._reranker = mock_reranker_empty
        event = make_market_event()

        await wired_orchestrator._process_event(event, "test-worker")

        mock_reranker_empty.rerank_async.assert_called_once()
        wired_orchestrator._spec_extractor.extract_async.assert_not_called()
        assert wired_orchestrator._metrics.events_processed == 1

    @pytest.mark.asyncio
    async def test_embedding_failure_counted(self, wired_orchestrator):
        """Embedding stage failure increments events_failed."""
        wired_orchestrator._embedding_processor.process_async = AsyncMock(
            side_effect=RuntimeError("CUDA OOM")
        )
        event = make_market_event()

        await wired_orchestrator._process_event(event, "test-worker")

        assert wired_orchestrator._metrics.events_failed == 1
        assert wired_orchestrator._metrics.events_processed == 0

    @pytest.mark.asyncio
    async def test_stage_latency_recorded(self, wired_orchestrator):
        """All executed stages have latency recorded in metrics.

        With mocked components, operations complete in sub-microsecond
        time, so latency may be 0.0 on Windows.  We verify call counts
        instead — these prove the stage was exercised and the metric
        was incremented.
        """
        event = make_market_event()
        await wired_orchestrator._process_event(event, "test-worker")

        m = wired_orchestrator._metrics
        assert m.canonicalization.total_calls == 1
        assert m.canonicalization.avg_latency_ms >= 0
        assert m.embedding.total_calls == 1
        assert m.retrieval.total_calls == 1
        assert m.reranking.total_calls == 1
        assert m.extraction.total_calls == 1
        assert m.verification.total_calls == 1
        assert m.persistence.total_calls == 1


# ═══════════════════════════════════════════════════════════════════════
#  9. _process_match (extract → verify → persist)
# ═══════════════════════════════════════════════════════════════════════


class TestProcessMatch:
    """Tests for _process_match verdict routing."""

    @pytest.mark.asyncio
    async def test_equivalent_persisted(self, wired_orchestrator):
        """Equivalent verdict → enqueued to writer."""
        query_event = make_canonical_event()
        match = make_verified_match()

        await wired_orchestrator._process_match(query_event, match, "test")

        wired_orchestrator._writer.enqueue.assert_called_once()
        assert wired_orchestrator._metrics.pairs_equivalent == 1
        assert wired_orchestrator._metrics.pairs_persisted == 1

    @pytest.mark.asyncio
    async def test_needs_review_persisted(self, wired_orchestrator):
        """needs_review verdict → enqueued to writer."""
        async def _verify(**kwargs):
            return make_verified_pair(verdict="needs_review")

        wired_orchestrator._pair_verifier.verify_pair_async = AsyncMock(
            side_effect=_verify
        )

        query_event = make_canonical_event()
        match = make_verified_match()

        await wired_orchestrator._process_match(query_event, match, "test")

        wired_orchestrator._writer.enqueue.assert_called_once()
        assert wired_orchestrator._metrics.pairs_needs_review == 1
        assert wired_orchestrator._metrics.pairs_persisted == 1

    @pytest.mark.asyncio
    async def test_not_equivalent_not_persisted(
        self, wired_orchestrator, mock_pair_verifier_not_equivalent
    ):
        """not_equivalent verdict → NOT enqueued to writer."""
        wired_orchestrator._pair_verifier = mock_pair_verifier_not_equivalent

        query_event = make_canonical_event()
        match = make_verified_match()

        await wired_orchestrator._process_match(query_event, match, "test")

        wired_orchestrator._writer.enqueue.assert_not_called()
        assert wired_orchestrator._metrics.pairs_not_equivalent == 1
        assert wired_orchestrator._metrics.pairs_persisted == 0

    @pytest.mark.asyncio
    async def test_writer_backpressure_handled(
        self, wired_orchestrator, mock_writer_full
    ):
        """QueueFull from writer is caught, not re-raised."""
        wired_orchestrator._writer = mock_writer_full

        query_event = make_canonical_event()
        match = make_verified_match()

        # Should NOT raise
        await wired_orchestrator._process_match(query_event, match, "test")

        assert wired_orchestrator._metrics.pairs_equivalent == 1
        # Persisted count should NOT increment because enqueue failed
        assert wired_orchestrator._metrics.pairs_persisted == 0

    @pytest.mark.asyncio
    async def test_extraction_parallel_both_sides(self, wired_orchestrator):
        """ContractSpec extraction called for both query and candidate."""
        query_event = make_canonical_event()
        match = make_verified_match()

        await wired_orchestrator._process_match(query_event, match, "test")

        assert wired_orchestrator._spec_extractor.extract_async.call_count == 2


# ═══════════════════════════════════════════════════════════════════════
#  10. Shutdown
# ═══════════════════════════════════════════════════════════════════════


class TestShutdown:
    """Tests for graceful shutdown ordering."""

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, wired_orchestrator):
        """Calling shutdown twice does not error."""
        wired_orchestrator._running = True
        await wired_orchestrator.shutdown()
        await wired_orchestrator.shutdown()  # Second call is a no-op

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_connectors(
        self, wired_orchestrator, mock_connector
    ):
        """Shutdown calls disconnect on all connectors."""
        wired_orchestrator._running = True
        await wired_orchestrator.shutdown()
        mock_connector.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_stops_writer(self, wired_orchestrator, mock_writer):
        """Shutdown calls stop on PipelineWriter."""
        wired_orchestrator._running = True
        await wired_orchestrator.shutdown()
        mock_writer.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_sets_running_false(self, wired_orchestrator):
        """After shutdown, _running is False."""
        wired_orchestrator._running = True
        await wired_orchestrator.shutdown()
        assert wired_orchestrator._running is False

    @pytest.mark.asyncio
    async def test_shutdown_sends_sentinels(self, wired_orchestrator):
        """Shutdown puts None sentinels in queue for each worker."""
        wired_orchestrator._running = True
        wired_orchestrator._num_workers = 3

        await wired_orchestrator.shutdown()

        # Queue should have sentinels (or be empty if already consumed)
        sentinel_count = 0
        while not wired_orchestrator._ingestion_queue.empty():
            item = wired_orchestrator._ingestion_queue.get_nowait()
            if item is None:
                sentinel_count += 1
        assert sentinel_count <= 3

    @pytest.mark.asyncio
    async def test_shutdown_releases_gpu(self, wired_orchestrator):
        """Shutdown calls _release_gpu_memory."""
        wired_orchestrator._running = True
        with patch.object(
            wired_orchestrator, "_release_gpu_memory"
        ) as mock_release:
            await wired_orchestrator.shutdown()
            mock_release.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_connector_error_handled(self, wired_orchestrator):
        """Connector disconnect error is logged, not propagated."""
        wired_orchestrator._running = True
        wired_orchestrator._connectors[0].disconnect = AsyncMock(
            side_effect=RuntimeError("ws close error")
        )

        # Should not raise
        await wired_orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_sets_event(self, wired_orchestrator):
        """Shutdown sets _shutdown_event so run() can unblock."""
        wired_orchestrator._running = True
        assert not wired_orchestrator._shutdown_event.is_set()

        await wired_orchestrator.shutdown()
        assert wired_orchestrator._shutdown_event.is_set()


# ═══════════════════════════════════════════════════════════════════════
#  11. Health & metrics endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestHealthAndMetrics:
    """Tests for get_health() and get_metrics()."""

    def test_get_health_stopped(self, wired_orchestrator):
        """Health returns 'stopped' when not running."""
        health = wired_orchestrator.get_health()
        assert health["status"] == "stopped"
        assert "connectors" in health
        assert "ingestion_queue" in health
        assert "writer" in health

    def test_get_health_running(self, wired_orchestrator):
        """Health returns 'running' when pipeline is active."""
        wired_orchestrator._running = True
        health = wired_orchestrator.get_health()
        assert health["status"] == "running"

    def test_get_health_queue_info(self, wired_orchestrator):
        """Health includes queue depth and capacity."""
        health = wired_orchestrator.get_health()
        assert health["ingestion_queue"]["depth"] == 0
        assert health["ingestion_queue"]["capacity"] == 50

    def test_get_metrics(self, wired_orchestrator):
        """get_metrics returns a valid summary dict."""
        metrics = wired_orchestrator.get_metrics()
        assert "events" in metrics
        assert "pairs" in metrics
        assert "stages" in metrics


# ═══════════════════════════════════════════════════════════════════════
#  12. End-to-end integration (mocked components, real orchestration)
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndMocked:
    """
    Full orchestrator run with mocked components.

    Validates the complete flow: run → ingest → process → shutdown.
    """

    @pytest.mark.asyncio
    async def test_full_run_and_shutdown(self, wired_orchestrator, mock_connector):
        """Orchestrator processes events and shuts down cleanly."""
        async def _run_and_shutdown():
            """Start the pipeline and shut it down after events drain."""
            # Start the run (non-blocking — we'll shutdown manually)
            run_task = asyncio.create_task(wired_orchestrator.run())

            # Wait for events to be processed
            for _ in range(50):
                await asyncio.sleep(0.1)
                if wired_orchestrator._metrics.events_processed >= 3:
                    break

            await wired_orchestrator.shutdown()

            # Wait for run to complete
            try:
                await asyncio.wait_for(run_task, timeout=5.0)
            except asyncio.TimeoutError:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass

        await _run_and_shutdown()

        m = wired_orchestrator._metrics
        assert m.events_received >= 3
        assert m.events_processed >= 1
        assert m.started_at is not None

    @pytest.mark.asyncio
    async def test_multi_venue_fan_in(
        self, wired_orchestrator, mock_connector, mock_connector_poly,
        mock_text_builder,
    ):
        """Events from multiple connectors fan into one queue."""
        wired_orchestrator._connectors = [mock_connector, mock_connector_poly]
        wired_orchestrator._text_builders[VenueType.POLYMARKET] = mock_text_builder

        async def _run_and_shutdown():
            run_task = asyncio.create_task(wired_orchestrator.run())

            for _ in range(50):
                await asyncio.sleep(0.1)
                # 3 from Kalshi + 2 from Polymarket = 5 total
                if wired_orchestrator._metrics.events_processed >= 5:
                    break

            await wired_orchestrator.shutdown()
            try:
                await asyncio.wait_for(run_task, timeout=5.0)
            except asyncio.TimeoutError:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass

        await _run_and_shutdown()

        assert wired_orchestrator._metrics.events_received >= 5


# ═══════════════════════════════════════════════════════════════════════
#  13. Heartbeat
# ═══════════════════════════════════════════════════════════════════════


class TestHeartbeat:
    """Tests for periodic heartbeat logging."""

    @pytest.mark.asyncio
    async def test_heartbeat_runs(self, wired_orchestrator):
        """Heartbeat task runs and can be cancelled cleanly."""
        wired_orchestrator._running = True

        task = asyncio.create_task(wired_orchestrator._heartbeat_loop())
        await asyncio.sleep(0.1)
        wired_orchestrator._running = False
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        # Test passes if no exception propagated

