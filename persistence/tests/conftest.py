"""Pytest configuration and fixtures for persistence tests."""

import asyncio
import json
import uuid
import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.types import MarketEvent, VenueType, EventType, OutcomeSpec
from canonicalization.types import CanonicalEvent
from canonicalization.contract_spec import ContractSpec, DateSpec, EntitySpec, ThresholdSpec
from matching.types import VerifiedPair
from persistence.writer import PairWriteRequest, PipelineWriter


# ── Market / Canonical Event fixtures ────────────────────────────────


@pytest.fixture
def market_event_a():
    """Sample MarketEvent for side A (Kalshi)."""
    return MarketEvent(
        venue=VenueType.KALSHI,
        venue_market_id="KXBTC-100K-2025",
        event_type=EventType.CREATED,
        title="Will Bitcoin reach $100k by 2025?",
        description="Bitcoin price market",
        outcomes=[
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No"),
        ],
        received_at=datetime.now(UTC),
    )


@pytest.fixture
def market_event_b():
    """Sample MarketEvent for side B (Polymarket)."""
    return MarketEvent(
        venue=VenueType.POLYMARKET,
        venue_market_id="poly-btc-100k-2025",
        event_type=EventType.CREATED,
        title="Bitcoin to $100,000 by end of 2025",
        description="Polymarket BTC price market",
        outcomes=[
            OutcomeSpec(outcome_id="yes", label="Yes"),
            OutcomeSpec(outcome_id="no", label="No"),
        ],
        received_at=datetime.now(UTC),
    )


@pytest.fixture
def canonical_event_a(market_event_a):
    """CanonicalEvent wrapping market A."""
    return CanonicalEvent(
        event=market_event_a,
        canonical_text="Market Statement:\nWill Bitcoin reach $100k by 2025?",
        content_hash="hash_a_content_001",
        identity_hash="hash_a_identity_001",
    )


@pytest.fixture
def canonical_event_b(market_event_b):
    """CanonicalEvent wrapping market B."""
    return CanonicalEvent(
        event=market_event_b,
        canonical_text="Market Statement:\nBitcoin to $100,000 by end of 2025",
        content_hash="hash_b_content_002",
        identity_hash="hash_b_identity_002",
    )


# ── ContractSpec fixtures ────────────────────────────────────────────


@pytest.fixture
def contract_spec_a():
    """ContractSpec for side A."""
    return ContractSpec(
        statement="Will Bitcoin reach $100k by 2025?",
        resolution_date=DateSpec(date=datetime(2025, 12, 31), is_deadline=True),
        entities=[
            EntitySpec(name="Bitcoin", entity_type="other", aliases=["BTC"]),
        ],
        thresholds=[
            ThresholdSpec(value=100000.0, unit="dollars", comparison=">="),
        ],
        resolution_criteria="Resolves Yes if BTC >= $100k on any major exchange",
        data_source="Coinbase",
        outcome_labels=["Yes", "No"],
        confidence=0.95,
    )


@pytest.fixture
def contract_spec_b():
    """ContractSpec for side B."""
    return ContractSpec(
        statement="Bitcoin to $100,000 by end of 2025",
        resolution_date=DateSpec(date=datetime(2025, 12, 31), is_deadline=True),
        entities=[
            EntitySpec(name="BTC", entity_type="other", aliases=["Bitcoin"]),
        ],
        thresholds=[
            ThresholdSpec(value=100000.0, unit="dollars", comparison=">="),
        ],
        resolution_criteria="Resolves based on BTC price on Coinbase",
        data_source="Coinbase",
        outcome_labels=["YES", "NO"],
        confidence=0.92,
    )


# ── VerifiedPair / PairWriteRequest fixtures ─────────────────────────


@pytest.fixture
def verified_pair(contract_spec_a, contract_spec_b):
    """Sample VerifiedPair for persistence tests."""
    return VerifiedPair(
        pair_key="kalshi:KXBTC-100K-2025::polymarket:poly-btc-100k-2025",
        market_a_id="KXBTC-100K-2025",
        market_b_id="poly-btc-100k-2025",
        contract_spec_a=contract_spec_a,
        contract_spec_b=contract_spec_b,
        outcome_mapping={"YES_A": "YES_B", "NO_A": "NO_B"},
        verdict="equivalent",
        confidence=0.93,
        comparison_details={
            "entity_score": 0.95,
            "threshold_score": 1.0,
            "date_score": 1.0,
            "data_source_match": True,
            "weighted_score": 0.93,
        },
    )


@pytest.fixture
def write_request(verified_pair, canonical_event_a, canonical_event_b):
    """Sample PairWriteRequest for persistence tests."""
    return PairWriteRequest(
        verified_pair=verified_pair,
        canonical_event_a=canonical_event_a,
        canonical_event_b=canonical_event_b,
        model_id="gpt-4o-mini",
        prompt_version="v1.0",
    )


def make_write_request(
    pair_key: str = "test-pair-key",
    venue_a: VenueType = VenueType.KALSHI,
    venue_market_id_a: str = "market-a-001",
    venue_b: VenueType = VenueType.POLYMARKET,
    venue_market_id_b: str = "market-b-001",
) -> PairWriteRequest:
    """
    Factory function for creating PairWriteRequest with custom fields.

    Useful for tests that need multiple distinct requests.

    Args:
        pair_key: Stable pair identifier.
        venue_a: Venue for side A.
        venue_market_id_a: Venue market ID for side A.
        venue_b: Venue for side B.
        venue_market_id_b: Venue market ID for side B.

    Returns:
        PairWriteRequest with the specified parameters.
    """
    event_a = MarketEvent(
        venue=venue_a,
        venue_market_id=venue_market_id_a,
        event_type=EventType.CREATED,
        title=f"Market A: {pair_key}",
        received_at=datetime.now(UTC),
    )
    event_b = MarketEvent(
        venue=venue_b,
        venue_market_id=venue_market_id_b,
        event_type=EventType.CREATED,
        title=f"Market B: {pair_key}",
        received_at=datetime.now(UTC),
    )
    canon_a = CanonicalEvent(
        event=event_a,
        canonical_text=f"Canonical A for {pair_key}",
        content_hash=f"hash-a-{pair_key}",
        identity_hash=f"id-a-{pair_key}",
    )
    canon_b = CanonicalEvent(
        event=event_b,
        canonical_text=f"Canonical B for {pair_key}",
        content_hash=f"hash-b-{pair_key}",
        identity_hash=f"id-b-{pair_key}",
    )
    spec_a = ContractSpec(
        statement=f"Statement A: {pair_key}",
        outcome_labels=["Yes", "No"],
        confidence=0.9,
    )
    spec_b = ContractSpec(
        statement=f"Statement B: {pair_key}",
        outcome_labels=["Yes", "No"],
        confidence=0.9,
    )
    pair = VerifiedPair(
        pair_key=pair_key,
        market_a_id=venue_market_id_a,
        market_b_id=venue_market_id_b,
        contract_spec_a=spec_a,
        contract_spec_b=spec_b,
        outcome_mapping={"YES_A": "YES_B", "NO_A": "NO_B"},
        verdict="equivalent",
        confidence=0.9,
        comparison_details={"weighted_score": 0.9},
    )
    return PairWriteRequest(
        verified_pair=pair,
        canonical_event_a=canon_a,
        canonical_event_b=canon_b,
        model_id="gpt-4o-mini",
        prompt_version="v1.0",
    )


# ── Mock asyncpg fixtures ────────────────────────────────────────────


@pytest.fixture
def mock_conn():
    """
    Mock asyncpg.Connection with fetchrow/execute as AsyncMock.

    Configures fetchrow to return a UUID-bearing Record-like dict by default.
    """
    conn = AsyncMock()
    test_uuid = str(uuid.uuid4())
    conn.fetchrow = AsyncMock(return_value={"id": test_uuid})
    conn.execute = AsyncMock()

    # Support `async with conn.transaction():`
    tx_ctx = AsyncMock()
    tx_ctx.__aenter__ = AsyncMock(return_value=tx_ctx)
    tx_ctx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx_ctx)

    return conn


@pytest.fixture
def mock_pool(mock_conn):
    """
    Mock asyncpg.Pool supporting `async with pool.acquire() as conn:`.
    """
    pool = AsyncMock()

    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=acquire_ctx)
    pool.close = AsyncMock()

    return pool


@pytest.fixture
def writer(mock_pool):
    """
    PipelineWriter with a pre-injected mock pool (skips real DB connect).
    """
    w = PipelineWriter(
        dsn="postgresql://test:test@localhost/test",
        batch_size=5,
        batch_timeout=0.3,
        queue_size=20,
        max_retries=3,
        notify_channel="pair_changes",
    )
    w._pool = mock_pool
    return w

