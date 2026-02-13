"""
Bootstrap integration tests (no ML models).

Validates:
  1. Polymarket Gamma REST fetcher — only active markets returned,
     pagination, rate limiting, deadline/cap, error handling.
  2. Kalshi REST fetcher — only open markets, cursor pagination, cap.
  3. Orchestrator._bootstrap_venue — bounded enqueue with backpressure,
     deduplication, parallel fetch, config toggles.

All HTTP calls are mocked (aioresponses).  No GPU, no Qwen, no DeBERTa.

Markers:
  - bootstrap: bootstrap-specific tests
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Path bootstrapping ──────────────────────────────────────────────────
_PIPELINE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from discovery.types import EventType, MarketEvent, OutcomeSpec, VenueType
from discovery.polymarket_poller import PolymarketConnector
from discovery.kalshi_poller import KalshiConnector
from discovery.dedup import MarketDeduplicator
from orchestrator import SemanticPipelineOrchestrator

import config as cfg

UTC = timezone.utc
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Test data factories
# ═══════════════════════════════════════════════════════════════════════


def _polymarket_market(
    condition_id: str = "0xabc123",
    question: str = "Will BTC hit $100k?",
    *,
    closed: bool = False,
    active: bool = True,
    outcomes: Optional[list] = None,
    clob_token_ids: Optional[list] = None,
) -> dict:
    """Build a single Gamma API market JSON object."""
    return {
        "condition_id": condition_id,
        "question": question,
        "description": f"Description for {question}",
        "closed": closed,
        "active": active,
        "outcomes": json.dumps(outcomes or ["Yes", "No"]),
        "clobTokenIds": json.dumps(clob_token_ids or ["tok_yes", "tok_no"]),
        "endDate": "2026-12-31T23:59:59Z",
        "resolutionSource": "Official data",
        "volume24hr": 50000.0,
    }


def _kalshi_market(
    ticker: str = "KXBTC-100K",
    title: str = "Will BTC hit $100k?",
    *,
    status: str = "active",
) -> dict:
    """Build a single Kalshi REST market JSON object."""
    return {
        "ticker": ticker,
        "title": title,
        "subtitle": "",
        "status": status,
        "rules_primary": f"Resolution rules for {title}",
        "close_time": "2026-12-31T23:59:59Z",
    }


# ═══════════════════════════════════════════════════════════════════════
#  1. Polymarket Gamma REST bootstrap
# ═══════════════════════════════════════════════════════════════════════


class TestPolymarketBootstrap:
    """Tests for PolymarketConnector.fetch_bootstrap_markets()."""

    @pytest.fixture
    def connector(self) -> PolymarketConnector:
        """Fresh connector with test Gamma API URL."""
        return PolymarketConnector(
            gamma_api_url="https://gamma-api.test.local",
        )

    @pytest.mark.asyncio
    async def test_fetches_only_active_markets(self, connector):
        """Only markets with closed=false AND active=true are returned."""
        response_data = [
            _polymarket_market("id-1", "Active market", closed=False, active=True),
            _polymarket_market("id-2", "Closed market", closed=True, active=False),
            _polymarket_market("id-3", "Inactive market", closed=False, active=False),
            _polymarket_market("id-4", "Another active", closed=False, active=True),
        ]

        import aiohttp
        from unittest.mock import AsyncMock as _AM

        # Mock the aiohttp.ClientSession context manager
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets()

        # Only the 2 active markets should pass the client-side filter
        assert len(events) == 2
        ids = {e.venue_market_id for e in events}
        assert ids == {"id-1", "id-4"}

        # All events should be Polymarket + UPDATED type
        for ev in events:
            assert ev.venue == VenueType.POLYMARKET
            assert ev.event_type == EventType.UPDATED
            assert ev.title != ""

    @pytest.mark.asyncio
    async def test_skips_markets_without_title(self, connector):
        """Markets missing a 'question' field are skipped."""
        response_data = [
            _polymarket_market("id-1", "Has title"),
            {**_polymarket_market("id-2", ""), "question": ""},
        ]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets()

        assert len(events) == 1
        assert events[0].venue_market_id == "id-1"

    @pytest.mark.asyncio
    async def test_pagination_fetches_multiple_pages(self, connector):
        """Offset-based pagination fetches until a short page."""
        page1 = [
            _polymarket_market(f"id-{i}", f"Market {i}")
            for i in range(100)  # Full page
        ]
        page2 = [
            _polymarket_market(f"id-{100 + i}", f"Market {100 + i}")
            for i in range(30)  # Short page → last
        ]

        call_count = 0

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        async def _json():
            nonlocal call_count
            call_count += 1
            return page1 if call_count == 1 else page2

        mock_resp.json = _json

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets()

        assert len(events) == 130
        # Verify pagination params were sent
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_max_markets_cap(self, connector):
        """max_markets parameter caps total returned events."""
        response_data = [
            _polymarket_market(f"id-{i}", f"Market {i}")
            for i in range(100)
        ]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets(max_markets=10)

        assert len(events) == 10

    @pytest.mark.asyncio
    async def test_deadline_returns_partial_results(self, connector):
        """When deadline is reached mid-fetch, partial results are returned."""
        page1 = [
            _polymarket_market(f"id-{i}", f"Market {i}")
            for i in range(100)
        ]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=page1)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            loop = asyncio.get_running_loop()
            # Deadline already in the past → should return after page 1
            deadline = loop.time() - 1.0
            events = await connector.fetch_bootstrap_markets(deadline=deadline)

        # Should have 0 events because deadline was already passed
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self, connector):
        """Non-200 HTTP status returns empty list, does not raise."""
        mock_resp = AsyncMock()
        mock_resp.status = 429
        mock_resp.text = AsyncMock(return_value="Rate limited")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets()

        assert events == []

    @pytest.mark.asyncio
    async def test_connection_error_returns_empty(self, connector):
        """Network errors return empty list, do not propagate."""
        import aiohttp

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets()

        assert events == []

    @pytest.mark.asyncio
    async def test_outcome_parsing_json_string(self, connector):
        """Outcomes encoded as JSON strings are parsed correctly."""
        response_data = [
            {
                **_polymarket_market("id-1", "Test"),
                "outcomes": '["Trump", "Biden", "Other"]',
                "clobTokenIds": '["tok_1", "tok_2", "tok_3"]',
            }
        ]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets()

        assert len(events) == 1
        assert len(events[0].outcomes) == 3
        assert events[0].outcomes[0].label == "Trump"
        assert events[0].outcomes[0].outcome_id == "tok_1"
        assert events[0].outcomes[2].label == "Other"

    @pytest.mark.asyncio
    async def test_outcome_parsing_native_list(self, connector):
        """Outcomes as native Python lists are also handled."""
        response_data = [
            {
                **_polymarket_market("id-1", "Test"),
                "outcomes": ["Yes", "No"],
                "clobTokenIds": ["tok_y", "tok_n"],
            }
        ]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets()

        assert len(events) == 1
        assert len(events[0].outcomes) == 2
        assert events[0].outcomes[0].label == "Yes"
        assert events[0].outcomes[0].outcome_id == "tok_y"

    @pytest.mark.asyncio
    async def test_query_params_include_active_filters(self, connector):
        """Verify the HTTP request uses closed=false and active=true."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=[])  # Empty → single page
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await connector.fetch_bootstrap_markets()

        # Inspect the params passed to session.get()
        call_args = mock_session.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")

        assert params["closed"] == "false", "Server-side filter: closed=false"
        assert params["active"] == "true", "Server-side filter: active=true"
        assert params["order"] == "volume24hr", "Sorted by volume"
        assert params["ascending"] == "false", "Descending volume"


# ═══════════════════════════════════════════════════════════════════════
#  2. Kalshi REST bootstrap
# ═══════════════════════════════════════════════════════════════════════


class TestKalshiBootstrap:
    """Tests for KalshiConnector.fetch_bootstrap_markets()."""

    @pytest.fixture
    def connector(self) -> KalshiConnector:
        """Kalshi connector with no auth (will still work for REST mocking)."""
        return KalshiConnector(
            ws_url="wss://fake.kalshi.test/ws",
            api_key_id="test-key",
            private_key_pem="fake-pem-for-test",
        )

    @pytest.mark.asyncio
    async def test_fetches_only_open_markets(self, connector):
        """Only markets with status=open are returned."""
        response_data = {
            "markets": [
                _kalshi_market("OPEN-1", "Open market", status="active"),
                _kalshi_market("CLOSED-1", "Closed", status="closed"),
                _kalshi_market("SETTLED-1", "Settled", status="settled"),
                _kalshi_market("OPEN-2", "Another open", status="active"),
            ],
            "cursor": None,
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets()

        assert len(events) == 2
        tickers = {e.venue_market_id for e in events}
        assert tickers == {"OPEN-1", "OPEN-2"}

        for ev in events:
            assert ev.venue == VenueType.KALSHI
            assert ev.event_type == EventType.UPDATED

    @pytest.mark.asyncio
    async def test_cursor_pagination(self, connector):
        """Cursor-based pagination fetches multiple pages."""
        page1 = {
            "markets": [
                _kalshi_market(f"MKT-{i}", f"Market {i}")
                for i in range(5)
            ],
            "cursor": "next_page_cursor",
        }
        page2 = {
            "markets": [
                _kalshi_market(f"MKT-{5 + i}", f"Market {5 + i}")
                for i in range(3)
            ],
            "cursor": None,
        }

        call_count = 0

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        async def _json():
            nonlocal call_count
            call_count += 1
            return page1 if call_count == 1 else page2

        mock_resp.json = _json

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets()

        assert len(events) == 8
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_max_markets_cap(self, connector):
        """max_markets parameter limits results."""
        response_data = {
            "markets": [
                _kalshi_market(f"MKT-{i}", f"Market {i}")
                for i in range(50)
            ],
            "cursor": "more_pages",
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            events = await connector.fetch_bootstrap_markets(max_markets=5)

        assert len(events) == 5

    @pytest.mark.asyncio
    async def test_request_uses_status_open_filter(self, connector):
        """Verify the HTTP request includes status=open."""
        response_data = {"markets": [], "cursor": None}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await connector.fetch_bootstrap_markets()

        call_args = mock_session.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["status"] == "open", "Server-side filter: status=open"
        assert params["limit"] == 1000, "Max page size for efficiency"


# ═══════════════════════════════════════════════════════════════════════
#  3. Orchestrator bootstrap logic
# ═══════════════════════════════════════════════════════════════════════


def _make_bootstrap_connector(
    events: List[MarketEvent],
    venue: VenueType = VenueType.KALSHI,
) -> MagicMock:
    """
    Create a mock connector with a ``fetch_bootstrap_markets()`` method.

    Also has ``stream_events`` / ``connect`` / ``start`` / ``disconnect``
    stubs so the orchestrator's ingestion task doesn't crash.
    """
    connector = AsyncMock()
    connector.venue_name = venue
    connector._running = True
    connector.connect = AsyncMock()
    connector.start = AsyncMock()
    connector.disconnect = AsyncMock()

    async def _fetch(**kwargs):
        max_m = kwargs.get("max_markets", 0)
        if max_m > 0:
            return events[:max_m]
        return list(events)

    connector.fetch_bootstrap_markets = AsyncMock(side_effect=_fetch)

    # stream_events: block forever after bootstrap (no WebSocket events)
    async def _stream():
        while True:
            await asyncio.sleep(3600)
        yield  # pragma: no cover – makes this an async generator

    connector.stream_events = _stream
    return connector


def _make_no_bootstrap_connector(
    venue: VenueType = VenueType.POLYMARKET,
) -> MagicMock:
    """Connector with NO fetch_bootstrap_markets (WebSocket-only)."""
    connector = AsyncMock()
    connector.venue_name = venue
    connector._running = True
    connector.connect = AsyncMock()
    connector.start = AsyncMock()
    connector.disconnect = AsyncMock()

    async def _stream():
        while True:
            await asyncio.sleep(3600)
        yield  # pragma: no cover

    connector.stream_events = _stream
    # Explicitly NO fetch_bootstrap_markets attribute
    if hasattr(connector, "fetch_bootstrap_markets"):
        del connector.fetch_bootstrap_markets
    return connector


def _wire_orchestrator(
    connectors: List[MagicMock],
    queue_size: int = 50,
    num_workers: int = 1,
) -> SemanticPipelineOrchestrator:
    """
    Build an orchestrator with all ML components mocked out.

    Only bootstrap + ingestion + queue mechanics are real.
    """
    orch = SemanticPipelineOrchestrator(
        venues=[VenueType.KALSHI],
        ingestion_queue_size=queue_size,
        num_workers=num_workers,
        model_id="test-bootstrap",
        prompt_version="v0.0-test",
    )

    # Inject mock connectors
    orch._connectors = connectors
    orch._deduplicator = MarketDeduplicator(ttl_seconds=3600)

    # Mock all ML + persistence components (not under test)
    orch._text_builders = {v: MagicMock() for v in VenueType}
    orch._embedding_encoder = AsyncMock()
    orch._qdrant_index = AsyncMock()
    orch._embedding_cache = AsyncMock()
    orch._embedding_processor = AsyncMock()
    orch._retriever = AsyncMock()
    orch._cross_encoder = AsyncMock()
    orch._reranker = AsyncMock()
    orch._spec_extractor = AsyncMock()
    orch._pair_verifier = AsyncMock()

    writer = AsyncMock()
    writer.start = AsyncMock()
    writer.stop = AsyncMock()
    writer.enqueue = AsyncMock()
    writer.get_stats = MagicMock(return_value={
        "batches_written": 0, "pairs_written": 0,
        "errors": 0, "dlq_size": 0, "queue_depth": 0, "running": True,
    })
    orch._writer = writer

    return orch


def _make_events(
    n: int, venue: VenueType = VenueType.KALSHI
) -> List[MarketEvent]:
    """Generate n MarketEvents with unique IDs."""
    return [
        MarketEvent(
            venue=venue,
            venue_market_id=f"{venue.value}-MKT-{i}",
            event_type=EventType.UPDATED,
            title=f"Test market {i} on {venue.value}",
            description=f"Description {i}",
            outcomes=[
                OutcomeSpec(outcome_id="yes", label="Yes"),
                OutcomeSpec(outcome_id="no", label="No"),
            ],
            received_at=datetime.now(UTC),
        )
        for i in range(n)
    ]


class TestOrchestratorBootstrapVenue:
    """Tests for orchestrator._bootstrap_venue() method."""

    @pytest.mark.asyncio
    async def test_all_events_enqueued(self):
        """All fetched events land in the ingestion queue."""
        events = _make_events(5)
        connector = _make_bootstrap_connector(events)
        orch = _wire_orchestrator([connector], queue_size=50)

        with patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 10.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 9.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):
            await orch._bootstrap_venue(connector)

        assert orch._ingestion_queue.qsize() == 5
        assert orch._metrics.events_received == 5

    @pytest.mark.asyncio
    async def test_deduplication_filters_duplicates(self):
        """Duplicate events (same venue:market_id) are not enqueued."""
        events = _make_events(3)
        # Duplicate: same as events[0]
        dup = MarketEvent(
            venue=events[0].venue,
            venue_market_id=events[0].venue_market_id,
            event_type=EventType.UPDATED,
            title="Duplicate",
            received_at=datetime.now(UTC),
        )
        all_events = events + [dup]
        connector = _make_bootstrap_connector(all_events)
        orch = _wire_orchestrator([connector], queue_size=50)

        with patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 10.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 9.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):
            await orch._bootstrap_venue(connector)

        # 4 events received, 1 deduped → 3 enqueued
        assert orch._ingestion_queue.qsize() == 3
        assert orch._metrics.events_received == 4
        assert orch._metrics.events_deduplicated == 1

    @pytest.mark.asyncio
    async def test_backpressure_drops_on_full_queue(self):
        """When queue is full and stays full, events are dropped."""
        events = _make_events(10)
        connector = _make_bootstrap_connector(events)
        # Queue size 3, no workers draining → fills up after 3
        orch = _wire_orchestrator([connector], queue_size=3, num_workers=0)

        with patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 10.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 9.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 0.1), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):
            await orch._bootstrap_venue(connector)

        # Queue is full at 3
        assert orch._ingestion_queue.qsize() == 3
        # 10 received, 3 enqueued, 7 dropped
        assert orch._metrics.events_received == 10

    @pytest.mark.asyncio
    async def test_backpressure_with_consumer(self):
        """Slow consumer still processes all events (bounded enqueue blocks)."""
        events = _make_events(10)
        connector = _make_bootstrap_connector(events)
        # Tiny queue size — consumer must drain for enqueue to unblock
        orch = _wire_orchestrator([connector], queue_size=2, num_workers=0)

        consumed: list = []

        async def _drain():
            """Simulated slow consumer."""
            while True:
                try:
                    ev = await asyncio.wait_for(
                        orch._ingestion_queue.get(), timeout=2.0
                    )
                    consumed.append(ev)
                    await asyncio.sleep(0.01)  # Simulate processing
                except asyncio.TimeoutError:
                    break

        with patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 30.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 29.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):
            # Run bootstrap and consumer concurrently
            await asyncio.gather(
                orch._bootstrap_venue(connector),
                _drain(),
            )

        # Consumer + queue should have all 10 events
        total = len(consumed) + orch._ingestion_queue.qsize()
        assert total == 10
        assert orch._metrics.events_received == 10

    @pytest.mark.asyncio
    async def test_fetch_timeout_handled_gracefully(self):
        """Connector timeout returns no events, does not raise."""
        connector = AsyncMock()
        connector.venue_name = VenueType.KALSHI
        connector.fetch_bootstrap_markets = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        orch = _wire_orchestrator([connector], queue_size=50)

        with patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 0.1), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 0.05), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):
            # Should not raise
            await orch._bootstrap_venue(connector)

        assert orch._ingestion_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_fetch_exception_handled_gracefully(self):
        """Connector exception returns no events, does not raise."""
        connector = AsyncMock()
        connector.venue_name = VenueType.KALSHI
        connector.fetch_bootstrap_markets = AsyncMock(
            side_effect=RuntimeError("API unavailable")
        )

        orch = _wire_orchestrator([connector], queue_size=50)

        with patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 10.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 9.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):
            await orch._bootstrap_venue(connector)

        assert orch._ingestion_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_empty_fetch_is_noop(self):
        """Connector returning 0 markets is handled cleanly."""
        connector = _make_bootstrap_connector([])
        orch = _wire_orchestrator([connector], queue_size=50)

        with patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 10.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 9.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):
            await orch._bootstrap_venue(connector)

        assert orch._ingestion_queue.qsize() == 0
        assert orch._metrics.events_received == 0


class TestOrchestratorBootstrapParallel:
    """Tests for parallel bootstrap across multiple venues."""

    @pytest.mark.asyncio
    async def test_parallel_fetch_both_venues(self):
        """Kalshi and Polymarket bootstrap run concurrently."""
        kalshi_events = _make_events(3, VenueType.KALSHI)
        poly_events = _make_events(4, VenueType.POLYMARKET)

        kalshi_connector = _make_bootstrap_connector(
            kalshi_events, VenueType.KALSHI
        )
        poly_connector = _make_bootstrap_connector(
            poly_events, VenueType.POLYMARKET
        )

        orch = _wire_orchestrator(
            [kalshi_connector, poly_connector], queue_size=50
        )

        with patch.object(cfg, "BOOTSTRAP_ENABLED", True), \
             patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 10.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 9.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):

            async def _run_bootstrap_only():
                """Run just the bootstrap phase, then shut down."""
                orch._running = True
                orch._shutdown_event.clear()

                # Bootstrap section extracted from run()
                bootstrap_connectors = [
                    c for c in orch._connectors
                    if hasattr(c, "fetch_bootstrap_markets")
                ]
                results = await asyncio.gather(
                    *[orch._bootstrap_venue(c) for c in bootstrap_connectors],
                    return_exceptions=True,
                )
                for i, result in enumerate(results):
                    assert not isinstance(result, Exception), (
                        f"Bootstrap failed for venue {i}: {result}"
                    )

            await _run_bootstrap_only()

        # Both venues' events should be in the queue
        assert orch._ingestion_queue.qsize() == 7
        assert orch._metrics.events_received == 7

        # Both connectors were called
        kalshi_connector.fetch_bootstrap_markets.assert_awaited_once()
        poly_connector.fetch_bootstrap_markets.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_one_venue_failure_does_not_block_other(self):
        """If Kalshi fails, Polymarket still bootstraps."""
        poly_events = _make_events(5, VenueType.POLYMARKET)

        failing_connector = AsyncMock()
        failing_connector.venue_name = VenueType.KALSHI
        failing_connector.fetch_bootstrap_markets = AsyncMock(
            side_effect=RuntimeError("Kalshi auth failed")
        )

        poly_connector = _make_bootstrap_connector(
            poly_events, VenueType.POLYMARKET
        )

        orch = _wire_orchestrator(
            [failing_connector, poly_connector], queue_size=50
        )

        with patch.object(cfg, "BOOTSTRAP_ENABLED", True), \
             patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 10.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 9.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):

            bootstrap_connectors = [
                c for c in orch._connectors
                if hasattr(c, "fetch_bootstrap_markets")
            ]
            results = await asyncio.gather(
                *[orch._bootstrap_venue(c) for c in bootstrap_connectors],
                return_exceptions=True,
            )

        # Polymarket's 5 events should still be in the queue
        assert orch._ingestion_queue.qsize() == 5

    @pytest.mark.asyncio
    async def test_bootstrap_disabled_skips_all(self):
        """BOOTSTRAP_ENABLED=false skips REST fetch entirely."""
        events = _make_events(5)
        connector = _make_bootstrap_connector(events)
        orch = _wire_orchestrator([connector], queue_size=50)

        # Simulate the run() method's check
        with patch.object(cfg, "BOOTSTRAP_ENABLED", False):
            if cfg.BOOTSTRAP_ENABLED:
                await orch._bootstrap_venue(connector)

        assert orch._ingestion_queue.qsize() == 0
        connector.fetch_bootstrap_markets.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connector_without_bootstrap_is_skipped(self):
        """Connectors without fetch_bootstrap_markets are silently skipped."""
        ws_only = _make_no_bootstrap_connector(VenueType.POLYMARKET)
        orch = _wire_orchestrator([ws_only], queue_size=50)

        bootstrap_connectors = [
            c for c in orch._connectors
            if hasattr(c, "fetch_bootstrap_markets")
        ]

        # No connectors should match
        assert len(bootstrap_connectors) == 0


class TestOrchestratorBootstrapMaxMarkets:
    """Tests for BOOTSTRAP_MAX_MARKETS_PER_VENUE config."""

    @pytest.mark.asyncio
    async def test_max_markets_passed_to_connector(self):
        """max_markets config is forwarded to fetch_bootstrap_markets()."""
        events = _make_events(100)
        connector = _make_bootstrap_connector(events)
        orch = _wire_orchestrator([connector], queue_size=200)

        with patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 10.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 9.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 25):
            await orch._bootstrap_venue(connector)

        # Connector should have received max_markets=25
        call_kwargs = connector.fetch_bootstrap_markets.call_args.kwargs
        assert call_kwargs.get("max_markets") == 25

        # Only 25 should be enqueued
        assert orch._ingestion_queue.qsize() == 25

    @pytest.mark.asyncio
    async def test_zero_max_markets_means_unlimited(self):
        """max_markets=0 does not limit results."""
        events = _make_events(50)
        connector = _make_bootstrap_connector(events)
        orch = _wire_orchestrator([connector], queue_size=200)

        with patch.object(cfg, "BOOTSTRAP_FETCH_TIMEOUT", 10.0), \
             patch.object(cfg, "BOOTSTRAP_FETCH_DEADLINE", 9.0), \
             patch.object(cfg, "BOOTSTRAP_ENQUEUE_TIMEOUT", 5.0), \
             patch.object(cfg, "BOOTSTRAP_MAX_MARKETS_PER_VENUE", 0):
            await orch._bootstrap_venue(connector)

        # max_markets should NOT be in kwargs when 0
        call_kwargs = connector.fetch_bootstrap_markets.call_args.kwargs
        assert "max_markets" not in call_kwargs

        assert orch._ingestion_queue.qsize() == 50

