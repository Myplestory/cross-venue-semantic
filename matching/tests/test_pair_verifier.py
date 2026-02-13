"""Unit tests for PairVerifier."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from matching.pair_verifier import PairVerifier, PairCache
from matching.types import VerifiedMatch, VerifiedPair, CandidateMatch
from canonicalization.contract_spec import ContractSpec


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_verifier_initialization_defaults():
    """Test PairVerifier initialization with default parameters."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    assert verifier._initialized is True
    assert verifier.cross_encoder_weight == 0.50
    assert verifier.threshold_weight == 0.20
    assert verifier.entity_weight == 0.15
    assert verifier.date_weight == 0.10
    assert verifier.data_source_weight == 0.05


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_verifier_initialization_custom_weights():
    """Test PairVerifier with custom weights."""
    verifier = PairVerifier(
        cross_encoder_weight=0.60,
        threshold_weight=0.25,
        entity_weight=0.10,
        date_weight=0.04,
        data_source_weight=0.01
    )
    await verifier.initialize()
    
    assert verifier.cross_encoder_weight == 0.60
    assert verifier.threshold_weight == 0.25
    total = (
        verifier.cross_encoder_weight + verifier.threshold_weight +
        verifier.entity_weight + verifier.date_weight + verifier.data_source_weight
    )
    assert abs(total - 1.0) < 1e-6


@pytest.mark.unit
def test_pair_verifier_weight_validation():
    """Test that weights must sum to 1.0."""
    # Valid weights
    verifier = PairVerifier(
        cross_encoder_weight=0.50,
        threshold_weight=0.20,
        entity_weight=0.15,
        date_weight=0.10,
        data_source_weight=0.05
    )
    assert verifier is not None
    
    # Invalid weights (should raise ValueError)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        PairVerifier(
            cross_encoder_weight=0.50,
            threshold_weight=0.20,
            entity_weight=0.15,
            date_weight=0.10,
            data_source_weight=0.10  # Sums to 1.05
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_verifier_cache_hit(sample_contract_spec_a, sample_contract_spec_b, sample_verified_match):
    """Test cache hit behavior."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    # Create a cached pair
    market_a_id = "market-a-123"
    market_b_id = "market-b-456"
    
    # First call (cache miss)
    result1 = await verifier.verify_pair_async(
        sample_verified_match,
        sample_contract_spec_a,
        sample_contract_spec_b,
        market_a_id,
        market_b_id
    )
    
    # Second call (cache hit)
    result2 = await verifier.verify_pair_async(
        sample_verified_match,
        sample_contract_spec_a,
        sample_contract_spec_b,
        market_a_id,
        market_b_id
    )
    
    # Results should be identical
    assert result1.pair_key == result2.pair_key
    assert result1.verdict == result2.verdict
    assert result1.confidence == result2.confidence


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_verifier_early_exit_low_cross_encoder():
    """Test early exit when cross-encoder score is too low."""
    verifier = PairVerifier(not_equivalent_threshold=0.5)
    await verifier.initialize()
    
    # Create a mock candidate match
    mock_candidate = MagicMock()
    mock_candidate.canonical_event = MagicMock()
    
    # Create VerifiedMatch with low score
    low_score_match = VerifiedMatch(
        candidate_match=mock_candidate,
        cross_encoder_score=0.3,  # Below threshold
        match_type="no_match",
        nli_scores={},
        primary_event_score=0.3
    )
    
    spec_a = ContractSpec(statement="Test A", outcome_labels=["Yes", "No"])
    spec_b = ContractSpec(statement="Test B", outcome_labels=["Yes", "No"])
    
    result = await verifier.verify_pair_async(
        low_score_match,
        spec_a,
        spec_b,
        "market-a",
        "market-b"
    )
    
    assert result.verdict == "not_equivalent"
    assert result.confidence == 0.3
    assert "early_exit" in result.comparison_details


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_verifier_binary_fast_path(sample_contract_spec_a, sample_contract_spec_b, sample_verified_match):
    """Test fast path for binary markets."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    # Ensure both specs are binary
    spec_a = ContractSpec(
        statement="Will Bitcoin reach $60k?",
        outcome_labels=["Yes", "No"]
    )
    spec_b = ContractSpec(
        statement="Will BTC reach $60k?",
        outcome_labels=["YES", "NO"]
    )
    
    result = await verifier.verify_pair_async(
        sample_verified_match,
        spec_a,
        spec_b,
        "market-a",
        "market-b"
    )
    
    # Should use fast path
    assert "fast_path" in result.comparison_details
    assert result.comparison_details["fast_path"] == "binary_market"
    assert len(result.outcome_mapping) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_verifier_critical_mismatch_early_exit(sample_verified_match):
    """Test early exit on critical entity mismatch."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    # Create specs with very different entities
    spec_a = ContractSpec(
        statement="Will Bitcoin reach $60k?",
        entities=[],  # No entities
        outcome_labels=["Yes", "No"]
    )
    spec_b = ContractSpec(
        statement="Will Ethereum reach $10k?",
        entities=[],  # No entities, but different statement
        outcome_labels=["Yes", "No"]
    )
    
    # Create a mock candidate match
    mock_candidate = MagicMock()
    mock_candidate.canonical_event = MagicMock()
    
    # Use a high cross-encoder score to avoid early exit
    high_score_match = VerifiedMatch(
        candidate_match=mock_candidate,
        cross_encoder_score=0.8,
        match_type="partial_match",
        nli_scores={},
        primary_event_score=0.8
    )
    
    result = await verifier.verify_pair_async(
        high_score_match,
        spec_a,
        spec_b,
        "market-a",
        "market-b"
    )
    
    # Should complete verification (not early exit on cross-encoder)
    assert result is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_verifier_weighted_score_calculation():
    """Test weighted score calculation."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    score = verifier._calculate_weighted_score(
        entity_score=0.8,
        threshold_score=0.9,
        date_score=0.95,
        data_source_score=1.0,
        cross_encoder_score=0.85
    )
    
    # Should be weighted combination
    expected = (
        0.15 * 0.8 +  # entity
        0.20 * 0.9 +  # threshold
        0.10 * 0.95 +  # date
        0.05 * 1.0 +  # data_source (matched)
        0.50 * 0.85   # cross_encoder (PRIMARY)
    )
    
    assert abs(score - expected) < 1e-6


@pytest.mark.unit
def test_pair_verifier_determine_verdict_equivalent():
    """Test verdict determination for equivalent pairs."""
    verifier = PairVerifier()
    
    verdict, confidence = verifier._determine_verdict(
        entity_score=0.9,
        threshold_score=0.9,
        date_score=0.9,
        weighted_score=0.95,
        cross_encoder_score=0.85
    )
    
    assert verdict == "equivalent"
    assert confidence == 0.95


@pytest.mark.unit
def test_pair_verifier_determine_verdict_not_equivalent():
    """Test verdict determination for non-equivalent pairs."""
    verifier = PairVerifier()
    
    verdict, confidence = verifier._determine_verdict(
        entity_score=0.2,  # Critical mismatch
        threshold_score=0.3,
        date_score=0.3,
        weighted_score=0.3,
        cross_encoder_score=0.3
    )
    
    assert verdict == "not_equivalent"
    assert confidence == 0.3


@pytest.mark.unit
def test_pair_verifier_determine_verdict_needs_review():
    """Test verdict determination for ambiguous pairs."""
    verifier = PairVerifier()
    
    verdict, confidence = verifier._determine_verdict(
        entity_score=0.7,
        threshold_score=0.7,
        date_score=0.7,
        weighted_score=0.7,  # Between thresholds
        cross_encoder_score=0.6
    )
    
    assert verdict == "needs_review"
    assert confidence == 0.7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_verifier_batch_processing(sample_verified_match):
    """Test batch processing of multiple pairs."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    # Create multiple pairs
    verified_matches = [sample_verified_match] * 3
    contract_specs = {}
    market_ids = []
    
    for i in range(3):
        spec_a = ContractSpec(
            statement=f"Test A {i}",
            outcome_labels=["Yes", "No"]
        )
        spec_b = ContractSpec(
            statement=f"Test B {i}",
            outcome_labels=["Yes", "No"]
        )
        
        market_a_id = f"market-a-{i}"
        market_b_id = f"market-b-{i}"
        
        contract_specs[f"{market_a_id}_spec"] = spec_a
        contract_specs[f"{market_b_id}_spec"] = spec_b
        market_ids.append((market_a_id, market_b_id))
    
    results = await verifier.verify_batch_async(
        verified_matches,
        contract_specs,
        market_ids
    )
    
    assert len(results) == 3
    assert all(isinstance(r, VerifiedPair) for r in results)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_verifier_generate_pair_key():
    """Test pair key generation (sorted)."""
    verifier = PairVerifier()
    
    key1 = verifier._generate_pair_key("market-a", "market-b")
    key2 = verifier._generate_pair_key("market-b", "market-a")
    
    # Should be the same regardless of order
    assert key1 == key2


@pytest.mark.unit
def test_pair_verifier_is_binary_market():
    """Test binary market detection."""
    verifier = PairVerifier()
    
    spec_a = ContractSpec(statement="Test", outcome_labels=["Yes", "No"])
    spec_b = ContractSpec(statement="Test", outcome_labels=["YES", "NO"])
    
    assert verifier._is_binary_market(spec_a, spec_b) is True
    
    spec_c = ContractSpec(statement="Test", outcome_labels=["A", "B", "C"])
    assert verifier._is_binary_market(spec_a, spec_c) is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pair_cache_operations():
    """Test PairCache operations."""
    cache = PairCache(max_size=10)
    await cache.initialize()
    
    # Create a test pair
    spec_a = ContractSpec(statement="Test A", outcome_labels=["Yes", "No"])
    spec_b = ContractSpec(statement="Test B", outcome_labels=["Yes", "No"])
    
    pair = VerifiedPair(
        pair_key="test-key",
        market_a_id="a",
        market_b_id="b",
        contract_spec_a=spec_a,
        contract_spec_b=spec_b,
        outcome_mapping={},
        verdict="equivalent",
        confidence=0.9,
        comparison_details={}
    )
    
    # Set and get
    await cache.set("test-key", pair)
    retrieved = await cache.get("test-key")
    
    assert retrieved is not None
    assert retrieved.pair_key == "test-key"
    
    # Cache miss
    assert await cache.get("non-existent") is None
    
    # Stats
    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1

