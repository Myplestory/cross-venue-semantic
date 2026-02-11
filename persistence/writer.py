"""
Database writer for semantic pipeline outputs.

Writes to:
- markets table (upsert by venue + venue_market_id)
- contract_specs table (idempotent by market_id + text_hash + model + prompt)
- verified_pairs table (transactional versioning with is_current flag)

Architecture:
- Bounded asyncio.Queue absorbs upstream writes without blocking the pipeline
- Background consumer drains in batches (size or timeout trigger)
- Each batch is a single Postgres transaction (atomicity)
- pg_notify('pair_changes', pair_key) wakes the Rust trading engine

Write order within each transaction:
1. Upsert markets (both sides) → get UUIDs
2. Upsert contract_specs (both sides) → get UUIDs
3. Version-insert verified_pairs (set old is_current=false, insert new)
4. pg_notify per pair_key
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

import asyncpg

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.types import CanonicalEvent
from canonicalization.contract_spec import ContractSpec
from matching.types import VerifiedPair
import config

logger = logging.getLogger(__name__)


@dataclass
class PairWriteRequest:
    """
    Bundles all data the writer needs for a single verified pair.

    The pipeline produces VerifiedPair objects with ContractSpec references,
    but the DB needs MarketEvent metadata (venue, venue_market_id, title) for
    the markets table.  PairWriteRequest bridges that gap.

    Attributes:
        verified_pair: The verified pair output from PairVerifier.
        canonical_event_a: CanonicalEvent for market A (contains MarketEvent + text + hash).
        canonical_event_b: CanonicalEvent for market B.
        model_id: LLM/extractor model identifier (for contract_specs.model_id).
        prompt_version: Extraction prompt version (for contract_specs.prompt_version).
        enqueued_at: Timestamp when this request was created.
    """

    verified_pair: VerifiedPair
    canonical_event_a: CanonicalEvent
    canonical_event_b: CanonicalEvent
    model_id: str
    prompt_version: str
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class PipelineWriter:
    """
    Nonblocking persistence writer for semantic pipeline outputs.

    Follows the same async queue-based pattern as EmbeddingProcessor:
    - Bounded asyncio.Queue provides backpressure
    - Background consumer drains batches (batch_size OR batch_timeout)
    - Single Postgres transaction per batch for atomicity
    - pg_notify('pair_changes') wakes the Rust data plane

    Write order within each transaction:
    1. Upsert markets (both sides) → get UUIDs
    2. Upsert contract_specs (both sides) → get UUIDs
    3. Version-insert verified_pair (set old is_current=false, insert new)
    4. pg_notify per pair_key

    Error handling:
    - Transient failures: exponential backoff retry (up to max_retries)
    - Persistent failures: dead-letter to in-memory DLQ with structured logging
    - Idempotent operations: ON CONFLICT DO UPDATE for markets, DO NOTHING for specs
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        batch_size: Optional[int] = None,
        batch_timeout: Optional[float] = None,
        queue_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        notify_channel: Optional[str] = None,
    ):
        """
        Initialize writer.

        Args:
            dsn: PostgreSQL connection string.  Falls back to config.DATABASE_URL.
            batch_size: Max items per batch before flush.  Default: 20.
            batch_timeout: Max seconds to wait before flushing partial batch.  Default: 2.0.
            queue_size: Bounded queue capacity (backpressure).  Default: 500.
            max_retries: Retry attempts for transient DB failures.  Default: 3.
            notify_channel: PostgreSQL NOTIFY channel name.  Default: 'pair_changes'.
        """
        self._dsn = dsn or config.DATABASE_URL
        self._batch_size = batch_size if batch_size is not None else config.WRITER_BATCH_SIZE
        self._batch_timeout = (
            batch_timeout if batch_timeout is not None else config.WRITER_BATCH_TIMEOUT
        )
        self._queue_size = queue_size if queue_size is not None else config.WRITER_QUEUE_SIZE
        self._max_retries = (
            max_retries if max_retries is not None else config.WRITER_MAX_RETRIES
        )
        self._notify_channel = notify_channel or config.WRITER_NOTIFY_CHANNEL

        # Internal state
        self._queue: asyncio.Queue[PairWriteRequest] = asyncio.Queue(
            maxsize=self._queue_size
        )
        self._pool: Optional[asyncpg.Pool] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._batches_written = 0
        self._pairs_written = 0
        self._errors = 0
        self._dlq: List[PairWriteRequest] = []

        logger.info(
            "PipelineWriter initialized: batch_size=%d, batch_timeout=%.1fs, "
            "queue_size=%d, max_retries=%d, channel=%s",
            self._batch_size,
            self._batch_timeout,
            self._queue_size,
            self._max_retries,
            self._notify_channel,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Create asyncpg connection pool.

        Raises:
            asyncpg.PostgresError: If connection to database fails.
        """
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=2,
            max_size=5,
            command_timeout=30.0,
        )
        logger.info("PipelineWriter pool created (min=2, max=5)")

    async def start(self) -> None:
        """Start background consumer task."""
        if self._consumer_task is not None and not self._consumer_task.done():
            return

        if self._pool is None:
            await self.initialize()

        self._running = True
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        logger.info("PipelineWriter consumer started")

    async def stop(self) -> None:
        """
        Graceful shutdown.

        Drains remaining queue items, writes final batch, closes pool.
        """
        self._running = False

        if self._consumer_task:
            # Give consumer time to drain remaining items
            try:
                await asyncio.wait_for(self._consumer_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass

        if self._pool:
            await self._pool.close()
            self._pool = None

        if self._dlq:
            logger.warning(
                "PipelineWriter stopped with %d items in DLQ", len(self._dlq)
            )

        logger.info(
            "PipelineWriter stopped: batches=%d, pairs=%d, errors=%d, dlq=%d",
            self._batches_written,
            self._pairs_written,
            self._errors,
            len(self._dlq),
        )

    # ── Public API ───────────────────────────────────────────────────────

    async def enqueue(self, request: PairWriteRequest) -> None:
        """
        Enqueue a verified pair for nonblocking persistence.

        Args:
            request: PairWriteRequest with verified pair and context.

        Raises:
            asyncio.QueueFull: If queue is at capacity (backpressure signal).
        """
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            logger.warning(
                "Writer queue full (%d items). Applying backpressure.",
                self._queue_size,
            )
            raise

    async def enqueue_batch(self, requests: List[PairWriteRequest]) -> int:
        """
        Enqueue multiple requests.  Returns count of successfully enqueued items.

        Args:
            requests: List of PairWriteRequest objects.

        Returns:
            Number of items successfully enqueued.
        """
        enqueued = 0
        for request in requests:
            try:
                self._queue.put_nowait(request)
                enqueued += 1
            except asyncio.QueueFull:
                logger.warning(
                    "Writer queue full after %d/%d items. Backpressure applied.",
                    enqueued,
                    len(requests),
                )
                break
        return enqueued

    def get_stats(self) -> Dict[str, Any]:
        """
        Return writer metrics.

        Returns:
            Dictionary with batches_written, pairs_written, errors,
            dlq_size, queue_depth, and running state.
        """
        return {
            "batches_written": self._batches_written,
            "pairs_written": self._pairs_written,
            "errors": self._errors,
            "dlq_size": len(self._dlq),
            "queue_depth": self._queue.qsize(),
            "running": self._running,
        }

    # ── Background consumer ──────────────────────────────────────────────

    async def _consumer_loop(self) -> None:
        """
        Background task: accumulate items into a batch, flush on size or timeout.

        Mirrors the EmbeddingProcessor._process_batches() pattern.
        """
        batch: List[PairWriteRequest] = []
        last_flush = datetime.now(UTC)

        while self._running or not self._queue.empty():
            try:
                # Try to get next item with short timeout
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=0.1
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass

                # Flush if batch is full or timeout exceeded
                elapsed = (datetime.now(UTC) - last_flush).total_seconds()
                if batch and (
                    len(batch) >= self._batch_size
                    or elapsed >= self._batch_timeout
                ):
                    await self._flush_batch_with_retry(batch)
                    batch = []
                    last_flush = datetime.now(UTC)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Unexpected error in consumer loop: %s", e, exc_info=True
                )
                await asyncio.sleep(0.5)

        # Drain remaining items on shutdown
        if batch:
            await self._flush_batch_with_retry(batch)

    async def _flush_batch_with_retry(
        self, batch: List[PairWriteRequest]
    ) -> None:
        """
        Flush batch with exponential backoff retry.

        On persistent failure after max_retries, items are moved to the DLQ.

        Args:
            batch: List of PairWriteRequest to persist.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                await self._flush_batch(batch)
                return
            except (
                asyncpg.PostgresConnectionError,
                asyncpg.InterfaceError,
                OSError,
            ) as e:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Transient DB error on attempt %d/%d, retrying in %ds: %s",
                    attempt,
                    self._max_retries,
                    wait,
                    e,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(wait)
            except Exception as e:
                logger.error(
                    "Non-retryable error writing batch of %d items: %s",
                    len(batch),
                    e,
                    exc_info=True,
                )
                break

        # Persistent failure → dead-letter
        self._errors += 1
        self._dlq.extend(batch)
        logger.error(
            "Batch of %d items moved to DLQ after %d attempts. DLQ size: %d",
            len(batch),
            self._max_retries,
            len(self._dlq),
        )

    # ── Transaction logic ────────────────────────────────────────────────

    async def _flush_batch(self, batch: List[PairWriteRequest]) -> None:
        """
        Write a batch to Postgres in a single transaction.

        Transaction order:
        1. Upsert markets (both sides) → UUIDs
        2. Upsert contract_specs (both sides) → UUIDs
        3. Version-insert verified_pairs (set old is_current=false, insert new)
        4. pg_notify('pair_changes') per pair_key

        Args:
            batch: List of PairWriteRequest to persist.

        Raises:
            asyncpg.PostgresError: On database errors (caller handles retry).
        """
        if not batch:
            return

        notified_keys: List[str] = []

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for request in batch:
                    pair = request.verified_pair

                    # 1. Upsert markets
                    market_a_uuid = await self._upsert_market(
                        conn, request.canonical_event_a
                    )
                    market_b_uuid = await self._upsert_market(
                        conn, request.canonical_event_b
                    )

                    # 2. Upsert contract_specs
                    spec_a_uuid = await self._upsert_contract_spec(
                        conn,
                        market_uuid=market_a_uuid,
                        text_hash=request.canonical_event_a.content_hash,
                        model_id=request.model_id,
                        prompt_version=request.prompt_version,
                        spec=pair.contract_spec_a,
                    )
                    spec_b_uuid = await self._upsert_contract_spec(
                        conn,
                        market_uuid=market_b_uuid,
                        text_hash=request.canonical_event_b.content_hash,
                        model_id=request.model_id,
                        prompt_version=request.prompt_version,
                        spec=pair.contract_spec_b,
                    )

                    # 3. Version-insert verified_pair
                    await self._insert_verified_pair(
                        conn,
                        pair=pair,
                        market_a_uuid=market_a_uuid,
                        market_b_uuid=market_b_uuid,
                        spec_a_uuid=spec_a_uuid,
                        spec_b_uuid=spec_b_uuid,
                    )

                    notified_keys.append(pair.pair_key)

                # 4. NOTIFY after all rows committed (still inside transaction)
                await self._notify_pair_changes(conn, notified_keys)

        self._batches_written += 1
        self._pairs_written += len(batch)
        logger.info(
            "Flushed batch: %d pairs written, %d notified. "
            "Total: batches=%d, pairs=%d",
            len(batch),
            len(notified_keys),
            self._batches_written,
            self._pairs_written,
        )

    # ── SQL operations ───────────────────────────────────────────────────

    async def _upsert_market(
        self, conn: asyncpg.Connection, event: CanonicalEvent
    ) -> str:
        """
        Upsert market row by (venue, venue_market_id).  Returns UUID PK.

        ON CONFLICT updates title, canonical_text, text_hash, and updated_at
        only if content has changed (text_hash differs).

        Args:
            conn: Active database connection (inside transaction).
            event: CanonicalEvent containing MarketEvent metadata.

        Returns:
            UUID primary key of the upserted market row (as string).
        """
        market = event.event
        row = await conn.fetchrow(
            """
            INSERT INTO markets (venue, venue_market_id, title, canonical_text, text_hash)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (venue, venue_market_id)
            DO UPDATE SET
                title = EXCLUDED.title,
                canonical_text = EXCLUDED.canonical_text,
                text_hash = EXCLUDED.text_hash,
                updated_at = now()
            WHERE markets.text_hash IS DISTINCT FROM EXCLUDED.text_hash
            RETURNING id
            """,
            market.venue.value,
            market.venue_market_id,
            market.title,
            event.canonical_text,
            event.content_hash,
        )

        if row:
            return str(row["id"])

        # ON CONFLICT matched but WHERE clause was false (no change) →
        # row exists with same text_hash, fetch its id
        existing = await conn.fetchrow(
            "SELECT id FROM markets WHERE venue = $1 AND venue_market_id = $2",
            market.venue.value,
            market.venue_market_id,
        )
        return str(existing["id"])

    async def _upsert_contract_spec(
        self,
        conn: asyncpg.Connection,
        market_uuid: str,
        text_hash: str,
        model_id: str,
        prompt_version: str,
        spec: ContractSpec,
    ) -> str:
        """
        Idempotent insert of contract_spec.  Returns UUID PK.

        Unique constraint: (market_id, text_hash, model_id, prompt_version).
        ON CONFLICT DO NOTHING — same extraction is never re-inserted.

        Args:
            conn: Active database connection.
            market_uuid: FK to markets.id.
            text_hash: Content hash of the canonical text at extraction time.
            model_id: LLM model identifier.
            prompt_version: Extraction prompt version.
            spec: ContractSpec Pydantic model.

        Returns:
            UUID primary key of the contract_spec row (as string).
        """
        spec_json = spec.model_dump_json()

        row = await conn.fetchrow(
            """
            INSERT INTO contract_specs (market_id, text_hash, model_id, prompt_version, spec_json)
            VALUES ($1::uuid, $2, $3, $4, $5::jsonb)
            ON CONFLICT (market_id, text_hash, model_id, prompt_version)
            DO NOTHING
            RETURNING id
            """,
            market_uuid,
            text_hash,
            model_id,
            prompt_version,
            spec_json,
        )

        if row:
            return str(row["id"])

        # Already existed → fetch id
        existing = await conn.fetchrow(
            """
            SELECT id FROM contract_specs
            WHERE market_id = $1::uuid
              AND text_hash = $2
              AND model_id = $3
              AND prompt_version = $4
            """,
            market_uuid,
            text_hash,
            model_id,
            prompt_version,
        )
        return str(existing["id"])

    async def _insert_verified_pair(
        self,
        conn: asyncpg.Connection,
        pair: VerifiedPair,
        market_a_uuid: str,
        market_b_uuid: str,
        spec_a_uuid: str,
        spec_b_uuid: str,
    ) -> str:
        """
        Version-insert verified pair with is_current flag management.

        Steps:
        1. Set is_current=false on all existing rows for this pair_key
        2. Insert new row with is_current=true

        This matches the schema's partial unique index:
        UNIQUE(pair_key) WHERE is_current = true

        Args:
            conn: Active database connection.
            pair: VerifiedPair from the pipeline.
            market_a_uuid: FK to markets.id for side A.
            market_b_uuid: FK to markets.id for side B.
            spec_a_uuid: FK to contract_specs.id for side A.
            spec_b_uuid: FK to contract_specs.id for side B.

        Returns:
            UUID primary key of the new verified_pair row (as string).
        """
        # Step 1: Retire previous version(s)
        await conn.execute(
            """
            UPDATE verified_pairs
            SET is_current = false, updated_at = now()
            WHERE pair_key = $1 AND is_current = true
            """,
            pair.pair_key,
        )

        # Step 2: Insert new current version
        outcome_json = json.dumps(pair.outcome_mapping)

        row = await conn.fetchrow(
            """
            INSERT INTO verified_pairs (
                pair_key, market_a_id, market_b_id,
                contract_spec_a_id, contract_spec_b_id,
                outcome_mapping, verdict,
                is_current, is_active
            ) VALUES (
                $1, $2::uuid, $3::uuid,
                $4::uuid, $5::uuid,
                $6::jsonb, $7::pair_verdict,
                true, true
            )
            RETURNING id
            """,
            pair.pair_key,
            market_a_uuid,
            market_b_uuid,
            spec_a_uuid,
            spec_b_uuid,
            outcome_json,
            pair.verdict,
        )

        return str(row["id"])

    async def _notify_pair_changes(
        self,
        conn: asyncpg.Connection,
        pair_keys: List[str],
    ) -> None:
        """
        Fire pg_notify for each changed pair.

        Payload is JSON with pair_key and timestamp.
        The Rust data plane LISTENs on this channel to refresh its pair config.

        Args:
            conn: Active database connection (inside transaction).
            pair_keys: List of pair_key values that were upserted.
        """
        for pair_key in pair_keys:
            payload = json.dumps({
                "pair_key": pair_key,
                "ts": datetime.now(UTC).isoformat(),
            })
            await conn.execute(
                "SELECT pg_notify($1, $2)",
                self._notify_channel,
                payload,
            )
