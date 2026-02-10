"""Performance tests for PairVerifier."""

import pytest
import time
from unittest.mock import MagicMock

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from matching.pair_verifier import PairVerifier
from matching.types import VerifiedMatch, CandidateMatch, VerifiedPair
from .fixtures.sample_contract_specs import (
    create_bitcoin_spec_60k,
    create_bitcoin_spec_80k
)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_cache_hit_performance():
    """Test cache hit latency (<1ms target)."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    spec_a = create_bitcoin_spec_60k()
    spec_b = create_bitcoin_spec_60k()
    
    # Create mock candidate match
    mock_candidate = MagicMock()
    mock_candidate.canonical_event = MagicMock()
    
    verified_match = VerifiedMatch(
        candidate_match=mock_candidate,
        cross_encoder_score=0.85,
        match_type="full_match",
        nli_scores={},
        primary_event_score=0.9
    )
    
    # First call (cache miss)
    start = time.perf_counter()
    result1 = await verifier.verify_pair_async(
        verified_match,
        spec_a,
        spec_b,
        "market-a",
        "market-b"
    )
    first_call_time = (time.perf_counter() - start) * 1000  # ms
    
    # Second call (cache hit)
    start = time.perf_counter()
    result2 = await verifier.verify_pair_async(
        verified_match,
        spec_a,
        spec_b,
        "market-a",
        "market-b"
    )
    cache_hit_time = (time.perf_counter() - start) * 1000  # ms
    
    assert cache_hit_time < 1.0, f"Cache hit took {cache_hit_time}ms, target <1ms"
    assert result1.pair_key == result2.pair_key


@pytest.mark.performance
@pytest.mark.asyncio
async def test_cache_miss_performance():
    """Test cache miss latency (<15ms target)."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    spec_a = create_bitcoin_spec_60k()
    spec_b = create_bitcoin_spec_80k()
    
    # Create mock candidate match
    mock_candidate = MagicMock()
    mock_candidate.canonical_event = MagicMock()
    
    verified_match = VerifiedMatch(
        candidate_match=mock_candidate,
        cross_encoder_score=0.85,
        match_type="full_match",
        nli_scores={},
        primary_event_score=0.9
    )
    
    start = time.perf_counter()
    result = await verifier.verify_pair_async(
        verified_match,
        spec_a,
        spec_b,
        "market-a",
        "market-b"
    )
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    assert elapsed < 15.0, f"Cache miss took {elapsed}ms, target <15ms"
    assert isinstance(result, VerifiedPair)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_binary_fast_path_performance():
    """Test binary market fast path latency (<5ms target)."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    spec_a = create_bitcoin_spec_60k()
    spec_b = create_bitcoin_spec_60k()
    
    # Create mock candidate match
    mock_candidate = MagicMock()
    mock_candidate.canonical_event = MagicMock()
    
    verified_match = VerifiedMatch(
        candidate_match=mock_candidate,
        cross_encoder_score=0.85,
        match_type="full_match",
        nli_scores={},
        primary_event_score=0.9
    )
    
    start = time.perf_counter()
    result = await verifier.verify_pair_async(
        verified_match,
        spec_a,
        spec_b,
        "market-a",
        "market-b"
    )
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    # Should use fast path for binary markets
    assert "fast_path" in result.comparison_details or elapsed < 5.0, \
        f"Binary fast path took {elapsed}ms, target <5ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_batch_throughput():
    """Test batch processing throughput (100+ pairs/sec target)."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    # Create 10 pairs
    verified_matches = []
    contract_specs = {}
    market_ids = []
    
    for i in range(10):
        spec_a = create_bitcoin_spec_60k()
        spec_b = create_bitcoin_spec_60k()
        
        market_a_id = f"market-a-{i}"
        market_b_id = f"market-b-{i}"
        
        contract_specs[f"{market_a_id}_spec"] = spec_a
        contract_specs[f"{market_b_id}_spec"] = spec_b
        market_ids.append((market_a_id, market_b_id))
        
        # Create mock candidate match
        mock_candidate = MagicMock()
        mock_candidate.canonical_event = MagicMock()
        
        verified_match = VerifiedMatch(
            candidate_match=mock_candidate,
            cross_encoder_score=0.85,
            match_type="full_match",
            nli_scores={},
            primary_event_score=0.9
        )
        verified_matches.append(verified_match)
    
    start = time.perf_counter()
    results = await verifier.verify_batch_async(
        verified_matches,
        contract_specs,
        market_ids
    )
    elapsed = time.perf_counter() - start
    
    throughput = len(results) / elapsed
    
    assert throughput >= 100.0, f"Throughput {throughput} pairs/sec, target >=100"
    assert len(results) == 10


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_verification():
    """Test concurrent verification requests."""
    import asyncio
    
    verifier = PairVerifier()
    await verifier.initialize()
    
    spec_a = create_bitcoin_spec_60k()
    spec_b = create_bitcoin_spec_60k()
    
    # Create mock candidate match
    mock_candidate = MagicMock()
    mock_candidate.canonical_event = MagicMock()
    
    verified_match = VerifiedMatch(
        candidate_match=mock_candidate,
        cross_encoder_score=0.85,
        match_type="full_match",
        nli_scores={},
        primary_event_score=0.9
    )
    
    # Create 10 concurrent requests
    tasks = [
        verifier.verify_pair_async(
            verified_match,
            spec_a,
            spec_b,
            f"market-a-{i}",
            f"market-b-{i}"
        )
        for i in range(10)
    ]
    
    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    
    assert len(results) == 10
    assert all(isinstance(r, VerifiedPair) for r in results)
    # Concurrent should be faster than sequential
    assert elapsed < 0.1, f"Concurrent verification took {elapsed}s"

