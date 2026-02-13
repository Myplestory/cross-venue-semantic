"""
Semantic Pipeline Orchestrator.

Coordinates the full pipeline:
Discovery → Canonicalization → Embedding → Retrieval → Reranking →
Extraction → Verification → Persistence

Architecture:
- Fan-in: Multiple venue connectors stream MarketEvents into a single
  bounded ingestion queue (backpressure via asyncio.Queue maxsize)
- N pipeline workers pull events and process through all stages
- GPU resource management via device configuration (config.py)
- Nonblocking persistence via PipelineWriter async queue
- Error isolation: per-event try/catch; failures logged, not propagated
- Graceful shutdown: signal-aware, drains queues before stopping

GPU Memory (8 GB RTX 3070):
- Recommended: Both quantized on CUDA:
    EMBEDDING_QUANTIZATION=true   (~4.5 GB)
    CROSS_ENCODER_QUANTIZATION=true (~0.5 GB)
    Total ≈ 6.5 GB — leaves ~1.5 GB for activations
- Alternative: CROSS_ENCODER_DEVICE=cpu (no quantization needed, ~30-70s/event reranking)
- 16 GB+ GPU: Both auto-detect to CUDA, quantization optional
- Orchestrator logs VRAM usage at startup for verification

Fintech standards applied:
- Structured logging with correlation IDs
- Heartbeat / periodic metrics reporting
- Connector auto-reconnection with exponential backoff
- GPU memory cleanup on shutdown
- Windows-safe signal handling (KeyboardInterrupt fallback)
"""

from __future__ import annotations

import asyncio
import gc
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path bootstrapping ──────────────────────────────────────────────────
# Allows ``from discovery.types import ...`` etc. when running from any cwd.
_PIPELINE_ROOT = str(Path(__file__).resolve().parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

# ── First-party imports (ordered by pipeline stage) ─────────────────────
from discovery.types import MarketEvent, VenueType          # noqa: E402
from discovery.venue_factory import create_connector        # noqa: E402
from discovery.base_connector import BaseVenueConnector     # noqa: E402
from discovery.dedup import MarketDeduplicator              # noqa: E402

from canonicalization.text_builder import (                 # noqa: E402
    get_builder,
    CanonicalTextBuilder,
)
from canonicalization.hasher import ContentHasher            # noqa: E402
from canonicalization.types import CanonicalEvent            # noqa: E402

from embedding.encoder import EmbeddingEncoder              # noqa: E402
from embedding.index import QdrantIndex                     # noqa: E402
from embedding.cache.in_memory import InMemoryCache         # noqa: E402
from embedding.processor import EmbeddingProcessor          # noqa: E402

from matching.retriever import CandidateRetriever           # noqa: E402
from matching.cross_encoder import CrossEncoder             # noqa: E402
from matching.reranker import CandidateReranker             # noqa: E402
from matching.pair_verifier import PairVerifier             # noqa: E402
from matching.types import VerifiedMatch, VerifiedPair      # noqa: E402

from extraction.spec_extractor import ContractSpecExtractor # noqa: E402

from persistence.writer import PipelineWriter, PairWriteRequest  # noqa: E402

import config                                               # noqa: E402

logger = logging.getLogger(__name__)

UTC = timezone.utc


# ═══════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class StageMetrics:
    """Per-stage latency and throughput tracker.

    All counters are only mutated from async coroutines on the same
    event-loop thread, so no lock is required.
    """

    total_calls: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    def record(self, latency_ms: float, *, error: bool = False) -> None:
        """Record a single stage execution."""
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms
        if latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if error:
            self.total_errors += 1

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe snapshot."""
        return {
            "calls": self.total_calls,
            "errors": self.total_errors,
            "avg_ms": round(self.avg_latency_ms, 1),
            "min_ms": round(self.min_latency_ms, 1) if self.total_calls else 0,
            "max_ms": round(self.max_latency_ms, 1),
        }


STAGE_NAMES = (
    "canonicalization",
    "embedding",
    "retrieval",
    "reranking",
    "extraction",
    "verification",
    "persistence",
)


@dataclass
class PipelineMetrics:
    """Aggregate pipeline metrics for health monitoring / Prometheus."""

    events_received: int = 0
    events_deduplicated: int = 0
    events_processed: int = 0
    events_failed: int = 0
    events_no_candidates: int = 0
    pairs_found: int = 0
    pairs_equivalent: int = 0
    pairs_needs_review: int = 0
    pairs_not_equivalent: int = 0
    pairs_persisted: int = 0
    started_at: Optional[datetime] = None

    # Per-stage
    canonicalization: StageMetrics = field(default_factory=StageMetrics)
    embedding: StageMetrics = field(default_factory=StageMetrics)
    retrieval: StageMetrics = field(default_factory=StageMetrics)
    reranking: StageMetrics = field(default_factory=StageMetrics)
    extraction: StageMetrics = field(default_factory=StageMetrics)
    verification: StageMetrics = field(default_factory=StageMetrics)
    persistence: StageMetrics = field(default_factory=StageMetrics)

    def summary(self) -> Dict[str, Any]:
        """Return metrics snapshot for health endpoint / logging."""
        uptime = 0.0
        if self.started_at:
            uptime = (datetime.now(UTC) - self.started_at).total_seconds()

        eps = (
            round(self.events_processed / (uptime / 60), 2)
            if uptime > 60
            else 0.0
        )

        return {
            "uptime_seconds": round(uptime, 1),
            "events": {
                "received": self.events_received,
                "deduplicated": self.events_deduplicated,
                "processed": self.events_processed,
                "failed": self.events_failed,
                "no_candidates": self.events_no_candidates,
            },
            "pairs": {
                "found": self.pairs_found,
                "equivalent": self.pairs_equivalent,
                "needs_review": self.pairs_needs_review,
                "not_equivalent": self.pairs_not_equivalent,
                "persisted": self.pairs_persisted,
            },
            "throughput_events_per_min": eps,
            "stages": {
                name: getattr(self, name).snapshot()
                for name in STAGE_NAMES
            },
        }


# ═══════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════


# Connector reconnection limits
_MAX_CONNECTOR_RETRIES = 10
_CONNECTOR_BASE_BACKOFF = 5.0   # seconds
_CONNECTOR_MAX_BACKOFF = 300.0  # 5 minutes cap

# Heartbeat interval
_HEARTBEAT_INTERVAL = 60.0  # seconds


class SemanticPipelineOrchestrator:
    """
    Top-level orchestrator for the semantic matching pipeline.

    Manages component lifecycles, coordinates per-event flow through
    all pipeline stages, and handles graceful shutdown.

    Usage::

        orchestrator = SemanticPipelineOrchestrator(
            venues=[VenueType.KALSHI, VenueType.POLYMARKET],
        )
        await orchestrator.initialize()
        await orchestrator.run()  # Blocks until SIGINT / SIGTERM
    """

    # ── Construction ─────────────────────────────────────────────────

    def __init__(
        self,
        venues: Optional[List[VenueType]] = None,
        ingestion_queue_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        model_id: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ):
        """
        Create orchestrator.  All parameters fall back to ``config.py``.

        Args:
            venues: Venue types to connect to.
            ingestion_queue_size: Bounded queue capacity (backpressure).
            num_workers: Concurrent pipeline workers (1 recommended for 8 GB GPU).
            model_id: Extraction model identifier (persisted to DB).
            prompt_version: Extraction prompt version (persisted to DB).
        """
        # ── Parse venues ────────────────────────────────────────────
        if venues is None:
            venue_str: str = config.ORCHESTRATOR_VENUES or ""
            venues = [
                VenueType(v.strip())
                for v in venue_str.split(",")
                if v.strip()
            ]
        self._venues: List[VenueType] = venues

        # ── Scalars ─────────────────────────────────────────────────
        self._ingestion_queue_size: int = (
            ingestion_queue_size
            if ingestion_queue_size is not None
            else config.ORCHESTRATOR_INGESTION_QUEUE_SIZE
        )
        self._num_workers: int = (
            num_workers
            if num_workers is not None
            else config.ORCHESTRATOR_NUM_WORKERS
        )
        self._model_id: str = model_id or config.ORCHESTRATOR_MODEL_ID
        self._prompt_version: str = prompt_version or config.ORCHESTRATOR_PROMPT_VERSION

        # ── Component slots (created in initialize) ─────────────────
        self._connectors: List[BaseVenueConnector] = []
        self._deduplicator: Optional[MarketDeduplicator] = None
        self._text_builders: Dict[VenueType, CanonicalTextBuilder] = {}
        self._embedding_encoder: Optional[EmbeddingEncoder] = None
        self._qdrant_index: Optional[QdrantIndex] = None
        self._embedding_cache: Optional[InMemoryCache] = None
        self._embedding_processor: Optional[EmbeddingProcessor] = None
        self._retriever: Optional[CandidateRetriever] = None
        self._cross_encoder: Optional[CrossEncoder] = None
        self._reranker: Optional[CandidateReranker] = None
        self._spec_extractor: Optional[ContractSpecExtractor] = None
        self._pair_verifier: Optional[PairVerifier] = None
        self._writer: Optional[PipelineWriter] = None

        # ── Internal state ──────────────────────────────────────────
        self._ingestion_queue: asyncio.Queue[Optional[MarketEvent]] = (
            asyncio.Queue(maxsize=self._ingestion_queue_size)
        )
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._connector_tasks: List[asyncio.Task] = []
        self._worker_tasks: List[asyncio.Task] = []
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._metrics: PipelineMetrics = PipelineMetrics()
        self._ingestion_log_count: int = 0

        logger.info(
            "Orchestrator created: venues=%s, workers=%d, queue_size=%d",
            [v.value for v in self._venues],
            self._num_workers,
            self._ingestion_queue_size,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ═══════════════════════════════════════════════════════════════════

    async def initialize(self) -> None:
        """
        Initialize all pipeline components in dependency order.

        GPU models are loaded eagerly so OOM / driver errors surface at
        startup rather than on the first event.

        Raises:
            RuntimeError: If no venue connectors could be created.
        """
        t_start = time.monotonic()
        logger.info("═══ Initializing semantic pipeline ═══")
        config.print_config_summary()

        # ── 1. Discovery ────────────────────────────────────────────
        self._deduplicator = MarketDeduplicator(
            ttl_seconds=config.ORCHESTRATOR_DEDUP_TTL,
        )

        for venue in self._venues:
            try:
                connector = create_connector(venue)
                self._connectors.append(connector)
                self._text_builders[venue] = get_builder(venue)
                logger.info("  ✓ Registered connector: %s", venue.value)
            except ValueError as exc:
                logger.warning(
                    "  ✗ Skipping unsupported venue %s: %s",
                    venue.value,
                    exc,
                )

        if not self._connectors:
            raise RuntimeError(
                "No venue connectors could be initialized.  "
                f"Requested: {[v.value for v in self._venues]}"
            )

        # ── 2. Embedding phase (GPU) ────────────────────────────────
        self._embedding_encoder = EmbeddingEncoder(
            model_name=config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            max_length=config.EMBEDDING_MAX_LENGTH,
            embedding_dim=config.EMBEDDING_DIM,
            instruction=config.EMBEDDING_INSTRUCTION,
            use_quantization=config.EMBEDDING_QUANTIZATION,
            gpu_concurrency=config.EMBEDDING_GPU_CONCURRENCY,
        )
        await self._embedding_encoder.initialize()
        logger.info(
            "  ✓ Embedding encoder on %s", self._embedding_encoder.device
        )

        self._qdrant_index = QdrantIndex(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            collection_name=config.QDRANT_COLLECTION_NAME,
            vector_size=config.QDRANT_VECTOR_SIZE,
        )
        await self._qdrant_index.initialize()
        logger.info("  ✓ Qdrant index ready")

        self._embedding_cache = InMemoryCache(
            max_size=config.EMBEDDING_CACHE_MAX_SIZE,
        )
        await self._embedding_cache.initialize()

        self._embedding_processor = EmbeddingProcessor(
            encoder=self._embedding_encoder,
            index=self._qdrant_index,
            cache=self._embedding_cache,
            batch_size=config.EMBEDDING_BATCH_SIZE,
        )
        # Sub-components already initialized individually above.
        # Calling processor.initialize() would redundantly reload the
        # encoder model — encoder's guard (`if self._initialized: return`)
        # prevents actual re-load but we skip it for clarity.

        # ── 3. Retrieval ────────────────────────────────────────────
        self._retriever = CandidateRetriever(index=self._qdrant_index)
        # Index already initialized above — retriever is ready.

        # ── 4. Matching phase (GPU or CPU) ──────────────────────────
        self._cross_encoder = CrossEncoder(
            model_name=config.CROSS_ENCODER_MODEL,
            device=config.CROSS_ENCODER_DEVICE,
            batch_size=config.CROSS_ENCODER_BATCH_SIZE,
            max_length=config.CROSS_ENCODER_MAX_LENGTH,
            use_quantization=config.CROSS_ENCODER_QUANTIZATION,
            gpu_concurrency=config.CROSS_ENCODER_GPU_CONCURRENCY,
        )
        await self._cross_encoder.initialize()
        logger.info(
            "  ✓ Cross-encoder on %s", self._cross_encoder.device
        )

        self._reranker = CandidateReranker(
            cross_encoder=self._cross_encoder,
            top_k=config.CROSS_ENCODER_TOP_K,
            score_threshold=config.CROSS_ENCODER_SCORE_THRESHOLD,
            primary_weight=config.CROSS_ENCODER_PRIMARY_WEIGHT,
            secondary_weight=config.CROSS_ENCODER_SECONDARY_WEIGHT,
        )

        # ── 5. Extraction ──────────────────────────────────────────
        self._spec_extractor = ContractSpecExtractor(
            use_llm_fallback=config.EXTRACTION_USE_LLM_FALLBACK,
            confidence_threshold=config.EXTRACTION_CONFIDENCE_THRESHOLD,
            high_confidence_threshold=config.EXTRACTION_HIGH_CONFIDENCE_THRESHOLD,
        )
        await self._spec_extractor.initialize()
        logger.info("  ✓ ContractSpec extractor ready")

        # ── 6. Verification ────────────────────────────────────────
        self._pair_verifier = PairVerifier()
        await self._pair_verifier.initialize()
        logger.info("  ✓ Pair verifier ready")

        # ── 7. Persistence ─────────────────────────────────────────
        self._writer = PipelineWriter(dsn=config.DATABASE_URL)
        await self._writer.initialize()
        logger.info("  ✓ Pipeline writer ready (DB pool open)")

        # ── GPU diagnostics ────────────────────────────────────────
        self._log_gpu_summary()

        elapsed = (time.monotonic() - t_start) * 1000
        logger.info(
            "═══ Semantic pipeline initialized in %.0f ms ═══", elapsed
        )

    # ─────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Start the pipeline and block until shutdown.

        Spawns:
        - One ingestion task per venue connector (fan-in to queue)
        - *N* pipeline worker tasks
        - PipelineWriter background consumer
        - Heartbeat / periodic metrics task

        On Unix, registers ``SIGINT`` / ``SIGTERM`` handlers.
        On Windows (where ``add_signal_handler`` raises
        ``NotImplementedError``), the caller should wrap this in
        ``try/except KeyboardInterrupt`` and call ``shutdown()``.
        """
        self._running = True
        self._shutdown_event.clear()
        self._metrics.started_at = datetime.now(UTC)

        # ── Signal handlers (Unix only) ─────────────────────────────
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(
                        self._signal_shutdown(s)
                    ),
                )
            except NotImplementedError:
                # Windows: rely on KeyboardInterrupt in __main__.py
                pass

        # ── Start persistence consumer ──────────────────────────────
        await self._writer.start()

        # ── Start pipeline workers before bootstrap ─────────────────
        # Workers must be running so bootstrap put() is consumed (backpressure);
        # otherwise the queue fills and we block on every put.
        for i in range(self._num_workers):
            task = asyncio.create_task(
                self._pipeline_worker(worker_id=i),
                name=f"worker-{i}",
            )
            self._worker_tasks.append(task)

        # ── Bootstrap: REST fetch of current markets (e.g. Kalshi) ───
        # Non-blocking put: put_nowait; on full queue yield to workers and retry once.
        # Hard timeout so a stuck REST call never blocks startup. Pass deadline so connector
        # can return partial results before we cancel (avoids losing all bootstrap data).
        _BOOTSTRAP_TIMEOUT = 90.0
        _BOOTSTRAP_DEADLINE_SEC = 85.0  # return partial results this many seconds in
        for connector in self._connectors:
            if not hasattr(connector, "fetch_bootstrap_markets"):
                continue
            venue = connector.venue_name.value
            try:
                logger.info("[%s] Bootstrap: fetching current markets (timeout %.0fs)...", venue, _BOOTSTRAP_TIMEOUT)
                loop = asyncio.get_event_loop()
                deadline = loop.time() + _BOOTSTRAP_DEADLINE_SEC
                bootstrap_events = await asyncio.wait_for(
                    connector.fetch_bootstrap_markets(deadline=deadline),
                    timeout=_BOOTSTRAP_TIMEOUT,
                )
                enqueued = 0
                dropped = 0
                for event in bootstrap_events:
                    self._metrics.events_received += 1
                    if self._deduplicator.is_duplicate(event):
                        self._metrics.events_deduplicated += 1
                        continue
                    try:
                        self._ingestion_queue.put_nowait(event)
                        enqueued += 1
                    except asyncio.QueueFull:
                        await asyncio.sleep(0.05)  # yield to workers
                        try:
                            self._ingestion_queue.put_nowait(event)
                            enqueued += 1
                        except asyncio.QueueFull:
                            dropped += 1
                if dropped:
                    logger.warning(
                        "[%s] Bootstrap: queue full, dropped %d market(s)",
                        venue,
                        dropped,
                    )
                if bootstrap_events:
                    logger.info(
                        "[%s] Bootstrap: enqueued %d/%d market(s) for pipeline",
                        venue,
                        enqueued,
                        len(bootstrap_events),
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] Bootstrap timed out after %.0fs — skipping REST bootstrap, WebSocket will stream updates",
                    venue,
                    _BOOTSTRAP_TIMEOUT,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Bootstrap failed: %s",
                    venue,
                    exc,
                    exc_info=True,
                )

        # ── Start venue connector ingestion tasks ───────────────────
        for connector in self._connectors:
            task = asyncio.create_task(
                self._ingest_from_connector(connector),
                name=f"ingest-{connector.venue_name.value}",
            )
            self._connector_tasks.append(task)

        # ── Start heartbeat ─────────────────────────────────────────
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="heartbeat"
        )

        logger.info(
            "Pipeline running: %d connector(s), %d worker(s)",
            len(self._connector_tasks),
            len(self._worker_tasks),
        )
        logger.info(
            "Ingestion: first 3 events per run logged; heartbeat every 60s shows rcvd= proc= pairs= queue="
        )

        # ── Block until shutdown completes ──────────────────────────
        await self._shutdown_event.wait()

    # ─────────────────────────────────────────────────────────────────

    async def _signal_shutdown(self, sig: signal.Signals) -> None:
        """Wrapper invoked by signal handler; calls ``shutdown()``."""
        logger.info("Received signal %s", sig.name)
        await self.shutdown()

    async def shutdown(self) -> None:
        """
        Graceful shutdown.

        Idempotent — safe to call multiple times.

        Order:
        1. Set ``_running = False`` → connectors + workers check this flag.
        2. Disconnect venue connectors (close WebSockets).
        3. Cancel connector tasks.
        4. Push sentinel ``None`` per worker → workers exit.
        5. Await worker drain with timeout.
        6. Cancel heartbeat.
        7. Stop ``PipelineWriter`` (drains its internal queue).
        8. Release GPU memory.
        9. Log final metrics.
        10. Set ``_shutdown_event`` → unblocks ``run()``.
        """
        if not self._running:
            return
        self._running = False
        logger.info("Shutdown initiated — draining pipeline...")

        # ── 1. Disconnect connectors ────────────────────────────────
        for connector in self._connectors:
            try:
                await connector.disconnect()
            except Exception as exc:
                logger.warning(
                    "Error disconnecting %s: %s",
                    connector.venue_name.value,
                    exc,
                )

        # ── 2. Cancel connector tasks ───────────────────────────────
        for task in self._connector_tasks:
            task.cancel()
        await asyncio.gather(*self._connector_tasks, return_exceptions=True)
        self._connector_tasks.clear()

        # ── 3. Send sentinel values to workers ──────────────────────
        for _ in range(self._num_workers):
            try:
                self._ingestion_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass  # Workers will also check _running flag

        # ── 4. Wait for workers to drain ────────────────────────────
        # The drain timeout must exceed the longest possible in-flight
        # GPU/CPU operation.  With Qwen3-4B INT8 cold-start (~60s) or
        # FP16 (~300s), and DeBERTa on CPU (~360s worst case), 600s
        # covers the extreme scenarios without force-cancelling.
        if self._worker_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *self._worker_tasks, return_exceptions=True
                    ),
                    timeout=600.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Worker drain timed out — force-cancelling")
                for task in self._worker_tasks:
                    task.cancel()
                await asyncio.gather(
                    *self._worker_tasks, return_exceptions=True
                )
            self._worker_tasks.clear()

        # ── 5. Cancel heartbeat ─────────────────────────────────────
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # ── 6. Stop persistence writer ──────────────────────────────
        if self._writer:
            await self._writer.stop()

        # ── 7. Release GPU memory ───────────────────────────────────
        self._release_gpu_memory()

        # ── 8. Final metrics ────────────────────────────────────────
        logger.info(
            "Pipeline stopped. Final metrics:\n%s",
            _format_metrics(self._metrics.summary()),
        )

        # ── 9. Unblock run() ────────────────────────────────────────
        self._shutdown_event.set()

    # ═══════════════════════════════════════════════════════════════════
    #  Ingestion (one task per connector)
    # ═══════════════════════════════════════════════════════════════════

    async def _ingest_from_connector(
        self, connector: BaseVenueConnector
    ) -> None:
        """
        Stream events from a venue connector into the ingestion queue.

        Implements auto-reconnection with exponential backoff.  A single
        connector failure does not affect other connectors or workers.

        Args:
            connector: Venue connector to stream from.
        """
        venue = connector.venue_name.value
        attempt = 0

        while self._running:
            try:
                # Connect + subscribe (WebSocket handshake)
                await connector.connect()
                await connector.start()
                attempt = 0  # reset on successful connection
                logger.info("[%s] Connected, streaming events", venue)

                async for event in connector.stream_events():
                    if not self._running:
                        return

                    self._metrics.events_received += 1

                    # Deduplication (fast, <1 ms)
                    if self._deduplicator.is_duplicate(event):
                        self._metrics.events_deduplicated += 1
                        continue

                    # Log first few enqueued events so user sees ingestion is working
                    if self._ingestion_log_count < 3:
                        logger.info(
                            "[%s] Ingesting event: %s — %s",
                            venue,
                            event.venue_market_id,
                            ((event.title or "").strip() or "(no title)")[:55],
                        )
                        self._ingestion_log_count += 1

                    # Enqueue with bounded wait (backpressure)
                    try:
                        await asyncio.wait_for(
                            self._ingestion_queue.put(event),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[%s] Queue full (%d/%d) — dropping %s",
                            venue,
                            self._ingestion_queue.qsize(),
                            self._ingestion_queue_size,
                            event.venue_market_id,
                        )

            except asyncio.CancelledError:
                logger.info("[%s] Ingestion task cancelled", venue)
                return

            except Exception as exc:
                if not self._running:
                    return

                attempt += 1
                if attempt > _MAX_CONNECTOR_RETRIES:
                    logger.error(
                        "[%s] Max reconnect attempts (%d) exceeded — "
                        "connector permanently stopped: %s",
                        venue,
                        _MAX_CONNECTOR_RETRIES,
                        exc,
                    )
                    return

                backoff = min(
                    _CONNECTOR_BASE_BACKOFF * (2 ** (attempt - 1)),
                    _CONNECTOR_MAX_BACKOFF,
                )
                logger.warning(
                    "[%s] Connection lost (attempt %d/%d): %s — "
                    "reconnecting in %.0fs",
                    venue,
                    attempt,
                    _MAX_CONNECTOR_RETRIES,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

            # Yield control between reconnection cycles — prevents CPU
            # starvation of other async tasks when the stream returns
            # immediately (e.g. empty stream, rapid disconnection).
            # In normal operation the WebSocket blocks inside the async-for
            # above, making this yield a no-op.  Cost: ~0.
            await asyncio.sleep(0)

    # ═══════════════════════════════════════════════════════════════════
    #  Pipeline worker (N concurrent tasks)
    # ═══════════════════════════════════════════════════════════════════

    async def _pipeline_worker(self, worker_id: int) -> None:
        """
        Pull events from ingestion queue and process through all stages.

        Each event is fully isolated: a failure in one event does not
        affect other events or workers.

        Args:
            worker_id: Worker index for structured logging.
        """
        tag = f"worker-{worker_id}"
        logger.info("[%s] Started", tag)

        while self._running or not self._ingestion_queue.empty():
            try:
                # Non-blocking get with short timeout so we can re-check
                # the _running flag periodically.
                try:
                    event: Optional[MarketEvent] = await asyncio.wait_for(
                        self._ingestion_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                # Sentinel = clean shutdown
                if event is None:
                    logger.debug("[%s] Received sentinel, exiting", tag)
                    break

                await self._process_event(event, tag)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                # Should never happen — _process_event has its own guard.
                logger.error(
                    "[%s] Unexpected worker-level error: %s",
                    tag,
                    exc,
                    exc_info=True,
                )
                self._metrics.events_failed += 1

        logger.info("[%s] Stopped", tag)

    # ═══════════════════════════════════════════════════════════════════
    #  Per-event pipeline
    # ═══════════════════════════════════════════════════════════════════

    async def _process_event(
        self, event: MarketEvent, tag: str
    ) -> None:
        """
        Process a single MarketEvent through all pipeline stages.

        Stages:
        1. Canonicalize          (CPU, <5 ms)
        2. Embed + Qdrant upsert (GPU + network)
        3. Retrieve candidates   (network)
        4. Rerank                (GPU / CPU)
        5–7. Per-match: Extract → Verify → Persist

        Args:
            event: MarketEvent from discovery.
            tag: Worker tag for structured logging.
        """
        event_id = f"{event.venue.value}:{event.venue_market_id}"
        pipeline_t0 = time.monotonic()

        try:
            # ── Stage 1: Canonicalization ────────────────────────────
            t0 = time.monotonic()
            canonical_event = self._canonicalize(event)
            self._metrics.canonicalization.record(
                (time.monotonic() - t0) * 1000
            )

            # ── Stage 2: Embedding ──────────────────────────────────
            t0 = time.monotonic()
            embedded_event = await self._embedding_processor.process_async(
                canonical_event
            )
            self._metrics.embedding.record(
                (time.monotonic() - t0) * 1000
            )

            # ── Stage 3: Retrieval ──────────────────────────────────
            t0 = time.monotonic()
            candidates = await self._retriever.retrieve_candidates(
                embedded_event,
                exclude_venue=event.venue,   # cross-venue matching only
            )
            self._metrics.retrieval.record(
                (time.monotonic() - t0) * 1000
            )

            if not candidates:
                self._metrics.events_no_candidates += 1
                self._metrics.events_processed += 1
                return

            # ── Stage 4: Reranking ──────────────────────────────────
            t0 = time.monotonic()
            verified_matches: List[VerifiedMatch] = (
                await self._reranker.rerank_async(
                    canonical_event, candidates
                )
            )
            self._metrics.reranking.record(
                (time.monotonic() - t0) * 1000
            )

            if not verified_matches:
                self._metrics.events_processed += 1
                return

            self._metrics.pairs_found += len(verified_matches)

            # ── Stages 5–7: per-match processing ────────────────────
            for match in verified_matches:
                try:
                    await self._process_match(
                        canonical_event, match, tag
                    )
                except Exception as exc:
                    logger.error(
                        "[%s] Match processing failed for %s: %s",
                        tag,
                        event_id,
                        exc,
                        exc_info=True,
                    )

            self._metrics.events_processed += 1

            elapsed_ms = (time.monotonic() - pipeline_t0) * 1000
            logger.info(
                "[%s] %s → %d candidates, %d matches, %.0f ms",
                tag,
                event_id,
                len(candidates),
                len(verified_matches),
                elapsed_ms,
            )

        except Exception as exc:
            self._metrics.events_failed += 1
            logger.error(
                "[%s] Pipeline failed for %s: %s",
                tag,
                event_id,
                exc,
                exc_info=True,
            )

    # ─────────────────────────────────────────────────────────────────

    async def _process_match(
        self,
        query_event: CanonicalEvent,
        match: VerifiedMatch,
        tag: str,
    ) -> None:
        """
        Extract ContractSpecs, verify the pair, and persist if actionable.

        Args:
            query_event: The query market's CanonicalEvent.
            match: VerifiedMatch from the reranking phase.
            tag: Worker tag for structured logging.
        """
        candidate_event: CanonicalEvent = (
            match.candidate_match.canonical_event
        )

        # ── Stage 5: ContractSpec extraction (parallel both sides) ──
        t0 = time.monotonic()
        spec_a, spec_b = await asyncio.gather(
            self._spec_extractor.extract_async(
                query_event.canonical_text,
                query_event.content_hash,
            ),
            self._spec_extractor.extract_async(
                candidate_event.canonical_text,
                candidate_event.content_hash,
            ),
        )
        self._metrics.extraction.record(
            (time.monotonic() - t0) * 1000
        )

        # ── Stage 6: Pair verification ──────────────────────────────
        t0 = time.monotonic()
        verified_pair: VerifiedPair = (
            await self._pair_verifier.verify_pair_async(
                verified_match=match,
                contract_spec_a=spec_a,
                contract_spec_b=spec_b,
                market_a_id=query_event.identity_hash,
                market_b_id=candidate_event.identity_hash,
            )
        )
        self._metrics.verification.record(
            (time.monotonic() - t0) * 1000
        )

        # ── Track verdict ───────────────────────────────────────────
        verdict = verified_pair.verdict
        if verdict == "equivalent":
            self._metrics.pairs_equivalent += 1
        elif verdict == "needs_review":
            self._metrics.pairs_needs_review += 1
        else:
            self._metrics.pairs_not_equivalent += 1

        # ── Stage 7: Persist (only actionable verdicts) ─────────────
        if verdict in ("equivalent", "needs_review"):
            t0 = time.monotonic()
            request = PairWriteRequest(
                verified_pair=verified_pair,
                canonical_event_a=query_event,
                canonical_event_b=candidate_event,
                model_id=self._model_id,
                prompt_version=self._prompt_version,
            )
            try:
                await self._writer.enqueue(request)
                self._metrics.pairs_persisted += 1
            except asyncio.QueueFull:
                logger.warning(
                    "[%s] Writer queue full — dropping pair %s",
                    tag,
                    verified_pair.pair_key,
                )
            self._metrics.persistence.record(
                (time.monotonic() - t0) * 1000
            )

    # ═══════════════════════════════════════════════════════════════════
    #  Canonicalization helper
    # ═══════════════════════════════════════════════════════════════════

    def _canonicalize(self, event: MarketEvent) -> CanonicalEvent:
        """
        Convert a raw ``MarketEvent`` into a ``CanonicalEvent``.

        Synchronous — CPU-bound, <5 ms.

        Args:
            event: Raw MarketEvent from discovery.

        Returns:
            CanonicalEvent with canonical text, content hash, and identity hash.

        Raises:
            ValueError: If no text builder is registered for the venue.
        """
        builder = self._text_builders.get(event.venue)
        if builder is None:
            raise ValueError(
                f"No text builder registered for venue: {event.venue.value}"
            )

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

    # ═══════════════════════════════════════════════════════════════════
    #  Heartbeat / periodic metrics
    # ═══════════════════════════════════════════════════════════════════

    async def _heartbeat_loop(self) -> None:
        """
        Log pipeline health every ``_HEARTBEAT_INTERVAL`` seconds.

        Also useful for liveness probes in Kubernetes / Railway.
        """
        try:
            while self._running:
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
                if not self._running:
                    break

                m = self._metrics
                queue_depth = self._ingestion_queue.qsize()

                logger.info(
                    "♥ HEARTBEAT | rcvd=%d proc=%d fail=%d "
                    "pairs=%d equiv=%d review=%d persisted=%d "
                    "queue=%d/%d",
                    m.events_received,
                    m.events_processed,
                    m.events_failed,
                    m.pairs_found,
                    m.pairs_equivalent,
                    m.pairs_needs_review,
                    m.pairs_persisted,
                    queue_depth,
                    self._ingestion_queue_size,
                )
        except asyncio.CancelledError:
            pass

    # ═══════════════════════════════════════════════════════════════════
    #  GPU diagnostics + cleanup
    # ═══════════════════════════════════════════════════════════════════

    def _log_gpu_summary(self) -> None:
        """Log GPU VRAM allocation summary at startup."""
        try:
            import torch
        except ImportError:
            logger.info("PyTorch not installed — GPU diagnostics skipped")
            return

        if not torch.cuda.is_available():
            logger.info("No CUDA GPU detected — running on CPU")
            return

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1e9
            alloc_gb = torch.cuda.memory_allocated(i) / 1e9
            logger.info(
                "  GPU %d (%s): %.1f GB total, %.1f GB allocated, "
                "%.1f GB free",
                i,
                props.name,
                total_gb,
                alloc_gb,
                total_gb - alloc_gb,
            )

        emb_dev = getattr(self._embedding_encoder, "device", "?")
        ce_dev = getattr(self._cross_encoder, "device", "?")
        logger.info(
            "  Model placement: embedding=%s  cross_encoder=%s",
            emb_dev,
            ce_dev,
        )

        if str(emb_dev) == "cuda" and str(ce_dev) == "cuda":
            total_mem_gb = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
            if total_mem_gb < 12:
                logger.warning(
                    "  ⚠ Both models on CUDA with only %.1f GB VRAM. "
                    "Set CROSS_ENCODER_DEVICE=cpu in .env if you hit OOM.",
                    total_mem_gb,
                )

    def _release_gpu_memory(self) -> None:
        """Release GPU memory on shutdown."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except ImportError:
            pass

        gc.collect()
        logger.debug("GC collect completed")

    # ═══════════════════════════════════════════════════════════════════
    #  Health / metrics (for HTTP endpoint or monitoring)
    # ═══════════════════════════════════════════════════════════════════

    def get_health(self) -> Dict[str, Any]:
        """
        Return pipeline health status for monitoring.

        Returns:
            Dictionary with component statuses and metrics.
        """
        return {
            "status": "running" if self._running else "stopped",
            "connectors": {
                c.venue_name.value: c._running
                for c in self._connectors
            },
            "ingestion_queue": {
                "depth": self._ingestion_queue.qsize(),
                "capacity": self._ingestion_queue_size,
            },
            "writer": self._writer.get_stats() if self._writer else {},
            "dedup_size": (
                self._deduplicator.size() if self._deduplicator else 0
            ),
            "metrics": self._metrics.summary(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return pipeline metrics summary."""
        return self._metrics.summary()


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _format_metrics(summary: Dict[str, Any]) -> str:
    """Pretty-format a metrics summary dict for logging."""
    lines = [
        f"  Uptime: {summary['uptime_seconds']}s",
        f"  Events: rcvd={summary['events']['received']} "
        f"proc={summary['events']['processed']} "
        f"fail={summary['events']['failed']} "
        f"dedup={summary['events']['deduplicated']} "
        f"no_cand={summary['events']['no_candidates']}",
        f"  Pairs:  found={summary['pairs']['found']} "
        f"equiv={summary['pairs']['equivalent']} "
        f"review={summary['pairs']['needs_review']} "
        f"not_equiv={summary['pairs']['not_equivalent']} "
        f"persisted={summary['pairs']['persisted']}",
        f"  Throughput: {summary['throughput_events_per_min']} events/min",
    ]
    for stage_name, stage_data in summary["stages"].items():
        lines.append(
            f"  {stage_name:20s} "
            f"calls={stage_data['calls']:>5d}  "
            f"errors={stage_data['errors']:>3d}  "
            f"avg={stage_data['avg_ms']:>7.1f}ms  "
            f"min={stage_data['min_ms']:>7.1f}ms  "
            f"max={stage_data['max_ms']:>7.1f}ms"
        )
    return "\n".join(lines)
