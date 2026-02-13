"""Unit tests for PipelineWriter.

Tests cover:
- Initialization and defaults
- Queue backpressure (bounded queue, QueueFull)
- Batch accumulation (size trigger, timeout trigger)
- Retry logic (exponential backoff on transient errors)
- Dead-letter queue (persistent failure handling)
- Transaction call order (markets → specs → pairs → notify)
- SQL operation logic (upsert fallback, idempotency, versioning)
- Graceful shutdown (drain queue before close)
- Metrics (get_stats accuracy)
"""

import asyncio
import json
import uuid
import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, call, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncpg
from persistence.writer import PipelineWriter, PairWriteRequest
from persistence.tests.conftest import make_write_request


# ═════════════════════════════════════════════════════════════════════
# Initialization
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_writer_initialization_defaults():
    """Test PipelineWriter initializes with config defaults."""
    writer = PipelineWriter(
        dsn="postgresql://test:test@localhost/test",
        batch_size=20,
        batch_timeout=2.0,
        queue_size=500,
        max_retries=3,
        notify_channel="pair_changes",
    )
    assert writer._batch_size == 20
    assert writer._batch_timeout == 2.0
    assert writer._queue_size == 500
    assert writer._max_retries == 3
    assert writer._notify_channel == "pair_changes"
    assert writer._running is False
    assert writer._pool is None
    assert writer._batches_written == 0
    assert writer._pairs_written == 0
    assert writer._errors == 0
    assert len(writer._dlq) == 0


@pytest.mark.unit
def test_writer_initialization_custom_params():
    """Test PipelineWriter honors custom parameters over config."""
    writer = PipelineWriter(
        dsn="postgresql://custom@host/db",
        batch_size=10,
        batch_timeout=0.5,
        queue_size=100,
        max_retries=5,
        notify_channel="custom_channel",
    )
    assert writer._dsn == "postgresql://custom@host/db"
    assert writer._batch_size == 10
    assert writer._batch_timeout == 0.5
    assert writer._queue_size == 100
    assert writer._max_retries == 5
    assert writer._notify_channel == "custom_channel"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_writer_initialize_creates_pool():
    """Test initialize() calls asyncpg.create_pool with correct params."""
    writer = PipelineWriter(
        dsn="postgresql://test@localhost/test",
        batch_size=5,
        batch_timeout=1.0,
        queue_size=10,
        max_retries=2,
    )

    mock_pool = AsyncMock()
    with patch(
        "persistence.writer.asyncpg.create_pool",
        new_callable=AsyncMock,
        return_value=mock_pool,
    ) as mock_create:
        await writer.initialize()

        mock_create.assert_called_once_with(
            dsn="postgresql://test@localhost/test",
            min_size=2,
            max_size=5,
            command_timeout=30.0,
        )
        assert writer._pool is mock_pool


@pytest.mark.unit
@pytest.mark.asyncio
async def test_writer_initialize_idempotent():
    """Test initialize() is a no-op if pool already exists."""
    writer = PipelineWriter(
        dsn="postgresql://test@localhost/test",
        batch_size=5,
        batch_timeout=1.0,
        queue_size=10,
        max_retries=2,
    )
    writer._pool = MagicMock()  # Simulate already initialized

    with patch("persistence.writer.asyncpg.create_pool") as mock_create:
        await writer.initialize()
        mock_create.assert_not_called()


# ═════════════════════════════════════════════════════════════════════
# Queue & Backpressure
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_single_item(writer, write_request):
    """Test enqueue() adds item to internal queue."""
    await writer.enqueue(write_request)
    assert writer._queue.qsize() == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_backpressure_raises_queue_full():
    """Test enqueue() raises QueueFull when queue is at capacity."""
    writer = PipelineWriter(
        dsn="postgresql://test@localhost/test",
        batch_size=5,
        batch_timeout=1.0,
        queue_size=3,
        max_retries=2,
    )

    # Fill the queue
    for i in range(3):
        req = make_write_request(pair_key=f"pair-{i}")
        await writer.enqueue(req)

    # Next enqueue should raise
    with pytest.raises(asyncio.QueueFull):
        await writer.enqueue(make_write_request(pair_key="overflow"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_batch_partial_success():
    """Test enqueue_batch() returns count of successfully enqueued items."""
    writer = PipelineWriter(
        dsn="postgresql://test@localhost/test",
        batch_size=5,
        batch_timeout=1.0,
        queue_size=3,
        max_retries=2,
    )

    requests = [make_write_request(pair_key=f"pair-{i}") for i in range(5)]
    enqueued = await writer.enqueue_batch(requests)

    assert enqueued == 3  # Queue capacity is 3
    assert writer._queue.qsize() == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_batch_all_success():
    """Test enqueue_batch() returns full count when queue has capacity."""
    writer = PipelineWriter(
        dsn="postgresql://test@localhost/test",
        batch_size=5,
        batch_timeout=1.0,
        queue_size=20,
        max_retries=2,
    )

    requests = [make_write_request(pair_key=f"pair-{i}") for i in range(5)]
    enqueued = await writer.enqueue_batch(requests)

    assert enqueued == 5
    assert writer._queue.qsize() == 5


# ═════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_get_stats_initial(writer):
    """Test get_stats() returns zeroed metrics on fresh writer."""
    stats = writer.get_stats()
    assert stats == {
        "batches_written": 0,
        "pairs_written": 0,
        "errors": 0,
        "dlq_size": 0,
        "queue_depth": 0,
        "running": False,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_stats_after_enqueue(writer, write_request):
    """Test get_stats() reflects queue depth after enqueue."""
    await writer.enqueue(write_request)
    stats = writer.get_stats()
    assert stats["queue_depth"] == 1
    assert stats["running"] is False


# ═════════════════════════════════════════════════════════════════════
# Transaction call order
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_flush_batch_call_order(writer, mock_conn, write_request):
    """Test _flush_batch() calls SQL operations in correct order."""
    await writer._flush_batch([write_request])

    # Verify fetchrow was called (for upsert_market A, upsert_market B,
    # upsert_spec A, upsert_spec B, insert_verified_pair)
    assert mock_conn.fetchrow.call_count == 5

    # Verify execute was called for:
    # - Retire old verified_pair (1 call)
    # - pg_notify (1 call per pair_key)
    assert mock_conn.execute.call_count == 2

    # Verify metrics updated
    assert writer._batches_written == 1
    assert writer._pairs_written == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_flush_batch_multiple_items(writer, mock_conn):
    """Test _flush_batch() processes all items in a batch."""
    requests = [make_write_request(pair_key=f"pair-{i}") for i in range(3)]
    await writer._flush_batch(requests)

    # 5 fetchrow calls per request (2 markets + 2 specs + 1 pair) × 3 = 15
    assert mock_conn.fetchrow.call_count == 15

    # 2 execute calls per request (1 retire + 1 notify) × 3... but notify is
    # done once per batch with all keys → 3 retire + 3 notify = 6
    # Actually: retire is per-request (3), notify is per pair_key (3)
    assert mock_conn.execute.call_count == 6

    assert writer._batches_written == 1
    assert writer._pairs_written == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_flush_batch_empty_is_noop(writer, mock_conn):
    """Test _flush_batch() with empty batch does nothing."""
    await writer._flush_batch([])

    mock_conn.fetchrow.assert_not_called()
    mock_conn.execute.assert_not_called()
    assert writer._batches_written == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_flush_batch_notifies_all_pair_keys(writer, mock_conn):
    """Test pg_notify fires for every pair_key in the batch."""
    requests = [make_write_request(pair_key=f"pair-{i}") for i in range(3)]
    await writer._flush_batch(requests)

    # Extract pg_notify calls (execute calls that contain "pg_notify")
    notify_calls = [
        c for c in mock_conn.execute.call_args_list
        if "pg_notify" in str(c)
    ]
    assert len(notify_calls) == 3

    # Verify each pair_key appears in a notify payload
    for i in range(3):
        payload_found = any(
            f"pair-{i}" in str(c) for c in notify_calls
        )
        assert payload_found, f"pair-{i} not found in notify calls"


# ═════════════════════════════════════════════════════════════════════
# SQL operation: market upsert
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_market_returns_uuid_on_insert(writer, mock_conn, canonical_event_a):
    """Test _upsert_market returns UUID from RETURNING clause on new insert."""
    expected_uuid = str(uuid.uuid4())
    mock_conn.fetchrow = AsyncMock(return_value={"id": expected_uuid})

    result = await writer._upsert_market(mock_conn, canonical_event_a)

    assert result == expected_uuid
    assert mock_conn.fetchrow.call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_market_fallback_select_on_no_change(
    writer, mock_conn, canonical_event_a
):
    """Test _upsert_market falls back to SELECT when text_hash unchanged."""
    fallback_uuid = str(uuid.uuid4())

    # First fetchrow returns None (ON CONFLICT WHERE was false → no row returned)
    # Second fetchrow returns the existing row
    mock_conn.fetchrow = AsyncMock(
        side_effect=[None, {"id": fallback_uuid}]
    )

    result = await writer._upsert_market(mock_conn, canonical_event_a)

    assert result == fallback_uuid
    assert mock_conn.fetchrow.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_market_passes_correct_params(
    writer, mock_conn, canonical_event_a
):
    """Test _upsert_market passes venue, venue_market_id, title, text, hash."""
    mock_conn.fetchrow = AsyncMock(return_value={"id": str(uuid.uuid4())})

    await writer._upsert_market(mock_conn, canonical_event_a)

    args = mock_conn.fetchrow.call_args
    # Positional args after SQL string
    sql_params = args[0][1:]
    assert sql_params[0] == "kalshi"  # venue.value
    assert sql_params[1] == "KXBTC-100K-2025"  # venue_market_id
    assert sql_params[2] == "Will Bitcoin reach $100k by 2025?"  # title
    assert "Market Statement:" in sql_params[3]  # canonical_text
    assert sql_params[4] == "hash_a_content_001"  # content_hash


# ═════════════════════════════════════════════════════════════════════
# SQL operation: contract spec upsert
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_spec_returns_uuid_on_insert(writer, mock_conn, contract_spec_a):
    """Test _upsert_contract_spec returns UUID on new insert."""
    expected_uuid = str(uuid.uuid4())
    mock_conn.fetchrow = AsyncMock(return_value={"id": expected_uuid})

    result = await writer._upsert_contract_spec(
        mock_conn,
        market_uuid=str(uuid.uuid4()),
        text_hash="hash_001",
        model_id="gpt-4o-mini",
        prompt_version="v1.0",
        spec=contract_spec_a,
    )

    assert result == expected_uuid
    assert mock_conn.fetchrow.call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_spec_fallback_on_conflict(writer, mock_conn, contract_spec_a):
    """Test _upsert_contract_spec falls back to SELECT on duplicate."""
    fallback_uuid = str(uuid.uuid4())

    # First call: ON CONFLICT DO NOTHING → returns None
    # Second call: SELECT existing → returns UUID
    mock_conn.fetchrow = AsyncMock(
        side_effect=[None, {"id": fallback_uuid}]
    )

    result = await writer._upsert_contract_spec(
        mock_conn,
        market_uuid=str(uuid.uuid4()),
        text_hash="hash_001",
        model_id="gpt-4o-mini",
        prompt_version="v1.0",
        spec=contract_spec_a,
    )

    assert result == fallback_uuid
    assert mock_conn.fetchrow.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_spec_serializes_pydantic_model(
    writer, mock_conn, contract_spec_a
):
    """Test _upsert_contract_spec serializes ContractSpec via model_dump_json."""
    mock_conn.fetchrow = AsyncMock(return_value={"id": str(uuid.uuid4())})

    await writer._upsert_contract_spec(
        mock_conn,
        market_uuid=str(uuid.uuid4()),
        text_hash="hash_001",
        model_id="gpt-4o-mini",
        prompt_version="v1.0",
        spec=contract_spec_a,
    )

    # The 5th positional arg after the SQL string is spec_json
    args = mock_conn.fetchrow.call_args[0]
    spec_json = args[5]

    # Verify it's valid JSON containing the statement
    parsed = json.loads(spec_json)
    assert parsed["statement"] == "Will Bitcoin reach $100k by 2025?"
    assert parsed["confidence"] == 0.95


# ═════════════════════════════════════════════════════════════════════
# SQL operation: verified pair versioning
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_pair_retires_old_version(writer, mock_conn, verified_pair):
    """Test _insert_verified_pair sets is_current=false on old rows first."""
    mock_conn.fetchrow = AsyncMock(return_value={"id": str(uuid.uuid4())})

    await writer._insert_verified_pair(
        mock_conn,
        pair=verified_pair,
        market_a_uuid=str(uuid.uuid4()),
        market_b_uuid=str(uuid.uuid4()),
        spec_a_uuid=str(uuid.uuid4()),
        spec_b_uuid=str(uuid.uuid4()),
    )

    # First call should be execute (UPDATE ... SET is_current = false)
    retire_call = mock_conn.execute.call_args_list[0]
    retire_sql = retire_call[0][0]
    assert "is_current = false" in retire_sql
    assert retire_call[0][1] == verified_pair.pair_key


@pytest.mark.unit
@pytest.mark.asyncio
async def test_insert_pair_inserts_new_current(writer, mock_conn, verified_pair):
    """Test _insert_verified_pair inserts new row with is_current=true."""
    expected_uuid = str(uuid.uuid4())
    mock_conn.fetchrow = AsyncMock(return_value={"id": expected_uuid})

    result = await writer._insert_verified_pair(
        mock_conn,
        pair=verified_pair,
        market_a_uuid=str(uuid.uuid4()),
        market_b_uuid=str(uuid.uuid4()),
        spec_a_uuid=str(uuid.uuid4()),
        spec_b_uuid=str(uuid.uuid4()),
    )

    assert result == expected_uuid

    # INSERT call should contain pair_key and verdict
    insert_call = mock_conn.fetchrow.call_args
    insert_sql = insert_call[0][0]
    assert "INSERT INTO verified_pairs" in insert_sql
    assert insert_call[0][1] == verified_pair.pair_key
    assert insert_call[0][6] == json.dumps(verified_pair.outcome_mapping)
    assert insert_call[0][7] == "equivalent"


# ═════════════════════════════════════════════════════════════════════
# SQL operation: pg_notify
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_fires_for_each_key(writer, mock_conn):
    """Test _notify_pair_changes fires pg_notify for each pair_key."""
    keys = ["pair-alpha", "pair-beta", "pair-gamma"]

    await writer._notify_pair_changes(mock_conn, keys)

    assert mock_conn.execute.call_count == 3

    for i, key in enumerate(keys):
        call_args = mock_conn.execute.call_args_list[i]
        assert call_args[0][0] == "SELECT pg_notify($1, $2)"
        assert call_args[0][1] == "pair_changes"

        payload = json.loads(call_args[0][2])
        assert payload["pair_key"] == key
        assert "ts" in payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_empty_keys_is_noop(writer, mock_conn):
    """Test _notify_pair_changes does nothing with empty list."""
    await writer._notify_pair_changes(mock_conn, [])
    mock_conn.execute.assert_not_called()


# ═════════════════════════════════════════════════════════════════════
# Retry logic
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retry_on_transient_connection_error(writer, write_request):
    """Test _flush_batch_with_retry retries on PostgresConnectionError."""
    call_count = 0

    async def fail_then_succeed(batch):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise asyncpg.PostgresConnectionError("connection lost")

    with patch.object(writer, "_flush_batch", side_effect=fail_then_succeed):
        with patch("persistence.writer.asyncio.sleep", new_callable=AsyncMock):
            await writer._flush_batch_with_retry([write_request])

    assert call_count == 3  # Failed 2 times, succeeded on 3rd
    assert writer._errors == 0
    assert len(writer._dlq) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retry_on_interface_error(writer, write_request):
    """Test _flush_batch_with_retry retries on InterfaceError."""
    call_count = 0

    async def fail_then_succeed(batch):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise asyncpg.InterfaceError("pool closed")

    with patch.object(writer, "_flush_batch", side_effect=fail_then_succeed):
        with patch("persistence.writer.asyncio.sleep", new_callable=AsyncMock):
            await writer._flush_batch_with_retry([write_request])

    assert call_count == 2
    assert writer._errors == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retry_on_os_error(writer, write_request):
    """Test _flush_batch_with_retry retries on OSError (network)."""
    call_count = 0

    async def fail_then_succeed(batch):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise OSError("ENETUNREACH")

    with patch.object(writer, "_flush_batch", side_effect=fail_then_succeed):
        with patch("persistence.writer.asyncio.sleep", new_callable=AsyncMock):
            await writer._flush_batch_with_retry([write_request])

    assert call_count == 2
    assert writer._errors == 0


# ═════════════════════════════════════════════════════════════════════
# Dead-letter queue
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dlq_on_persistent_transient_error(writer, write_request):
    """Test items move to DLQ after max_retries exhausted."""

    async def always_fail(batch):
        raise asyncpg.PostgresConnectionError("connection lost")

    with patch.object(writer, "_flush_batch", side_effect=always_fail):
        with patch("persistence.writer.asyncio.sleep", new_callable=AsyncMock):
            await writer._flush_batch_with_retry([write_request])

    assert writer._errors == 1
    assert len(writer._dlq) == 1
    assert writer._dlq[0] is write_request


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dlq_on_non_retryable_error(writer, write_request):
    """Test non-retryable errors go to DLQ immediately (no retry)."""

    async def raise_value_error(batch):
        raise ValueError("corrupt data")

    with patch.object(writer, "_flush_batch", side_effect=raise_value_error):
        await writer._flush_batch_with_retry([write_request])

    # Should only attempt once (non-retryable breaks immediately)
    assert writer._errors == 1
    assert len(writer._dlq) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dlq_accumulates_across_batches(writer):
    """Test DLQ accumulates items from multiple failed batches."""

    async def always_fail(batch):
        raise asyncpg.PostgresConnectionError("connection lost")

    with patch.object(writer, "_flush_batch", side_effect=always_fail):
        with patch("persistence.writer.asyncio.sleep", new_callable=AsyncMock):
            req1 = make_write_request(pair_key="fail-1")
            req2 = make_write_request(pair_key="fail-2")

            await writer._flush_batch_with_retry([req1])
            await writer._flush_batch_with_retry([req2])

    assert writer._errors == 2
    assert len(writer._dlq) == 2


# ═════════════════════════════════════════════════════════════════════
# Consumer loop: batch accumulation
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consumer_flushes_on_batch_size(writer, mock_conn):
    """Test consumer loop flushes when batch reaches batch_size items."""
    # batch_size is 5 for the writer fixture
    for i in range(5):
        await writer.enqueue(make_write_request(pair_key=f"pair-{i}"))

    writer._running = True

    flush_called = asyncio.Event()
    original_flush = writer._flush_batch_with_retry

    async def track_flush(batch):
        await original_flush(batch)
        flush_called.set()
        writer._running = False  # Stop after first flush

    with patch.object(writer, "_flush_batch_with_retry", side_effect=track_flush):
        task = asyncio.create_task(writer._consumer_loop())

        try:
            await asyncio.wait_for(flush_called.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            writer._running = False

        await asyncio.wait_for(task, timeout=2.0)

    assert writer._queue.empty()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consumer_flushes_on_timeout(writer, mock_conn):
    """Test consumer loop flushes partial batch after batch_timeout."""
    # batch_timeout is 0.3s for the writer fixture, batch_size is 5
    # Enqueue only 2 items (below batch_size), should flush on timeout
    for i in range(2):
        await writer.enqueue(make_write_request(pair_key=f"pair-{i}"))

    writer._running = True

    flush_called = asyncio.Event()

    async def track_flush(batch):
        assert len(batch) == 2  # Should have the 2 items
        flush_called.set()
        writer._running = False

    with patch.object(writer, "_flush_batch_with_retry", side_effect=track_flush):
        task = asyncio.create_task(writer._consumer_loop())

        try:
            await asyncio.wait_for(flush_called.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            writer._running = False

        await asyncio.wait_for(task, timeout=2.0)

    assert writer._queue.empty()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consumer_drains_on_shutdown(writer, mock_conn):
    """Test consumer drains remaining items when _running set to False."""
    for i in range(3):
        await writer.enqueue(make_write_request(pair_key=f"pair-{i}"))

    writer._running = False  # Already stopped

    flushed_count = 0

    async def track_flush(batch):
        nonlocal flushed_count
        flushed_count += len(batch)

    with patch.object(writer, "_flush_batch_with_retry", side_effect=track_flush):
        await writer._consumer_loop()

    assert flushed_count == 3
    assert writer._queue.empty()


# ═════════════════════════════════════════════════════════════════════
# Lifecycle: start / stop
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_creates_consumer_task(writer):
    """Test start() creates a background consumer task."""
    writer._running = False

    await writer.start()

    assert writer._running is True
    assert writer._consumer_task is not None
    assert not writer._consumer_task.done()

    # Cleanup
    writer._running = False
    writer._consumer_task.cancel()
    try:
        await writer._consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_is_idempotent(writer):
    """Test start() is a no-op if already running."""
    await writer.start()
    first_task = writer._consumer_task

    await writer.start()
    assert writer._consumer_task is first_task  # Same task

    # Cleanup
    writer._running = False
    writer._consumer_task.cancel()
    try:
        await writer._consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_closes_pool(writer, mock_pool):
    """Test stop() closes the asyncpg pool."""
    writer._running = True

    await writer.stop()

    assert writer._running is False
    mock_pool.close.assert_awaited_once()
    assert writer._pool is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_logs_dlq_warning(writer, mock_pool, write_request):
    """Test stop() logs warning when DLQ is non-empty."""
    writer._dlq.append(write_request)

    await writer.stop()

    stats = writer.get_stats()
    assert stats["dlq_size"] == 1


# ═════════════════════════════════════════════════════════════════════
# PairWriteRequest dataclass
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_pair_write_request_has_enqueued_at(write_request):
    """Test PairWriteRequest auto-sets enqueued_at timestamp."""
    assert write_request.enqueued_at is not None
    assert isinstance(write_request.enqueued_at, datetime)


@pytest.mark.unit
def test_pair_write_request_fields(write_request):
    """Test PairWriteRequest carries all required fields."""
    assert write_request.verified_pair is not None
    assert write_request.canonical_event_a is not None
    assert write_request.canonical_event_b is not None
    assert write_request.model_id == "gpt-4o-mini"
    assert write_request.prompt_version == "v1.0"


@pytest.mark.unit
def test_make_write_request_factory():
    """Test make_write_request factory creates distinct requests."""
    req1 = make_write_request(pair_key="pair-1", venue_market_id_a="m1-a")
    req2 = make_write_request(pair_key="pair-2", venue_market_id_a="m2-a")

    assert req1.verified_pair.pair_key != req2.verified_pair.pair_key
    assert req1.canonical_event_a.event.venue_market_id != (
        req2.canonical_event_a.event.venue_market_id
    )


# ═════════════════════════════════════════════════════════════════════
# End-to-end: enqueue → consumer → flush
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.asyncio
async def test_end_to_end_enqueue_start_stop(writer, mock_conn):
    """Test full lifecycle: enqueue items → start → stop → verify flushed."""
    for i in range(3):
        await writer.enqueue(make_write_request(pair_key=f"e2e-{i}"))

    await writer.start()

    # Allow consumer to process
    await asyncio.sleep(0.5)

    await writer.stop()

    stats = writer.get_stats()
    assert stats["queue_depth"] == 0
    assert stats["pairs_written"] == 3
    assert stats["batches_written"] >= 1
    assert stats["errors"] == 0
    assert stats["dlq_size"] == 0

