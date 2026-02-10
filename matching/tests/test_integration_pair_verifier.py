"""Integration tests for PairVerifier with golden dataset."""

import pytest
from datetime import datetime

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from matching.pair_verifier import PairVerifier
from matching.types import VerifiedMatch, CandidateMatch, VerifiedPair
from canonicalization.contract_spec import ContractSpec
from .fixtures.golden_dataset import get_golden_dataset, get_edge_case_dataset


@pytest.mark.integration
@pytest.mark.asyncio
async def test_golden_dataset_equivalent_pairs():
    """Test verification against known equivalent pairs."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    golden_dataset = get_golden_dataset()
    equivalent_cases = [c for c in golden_dataset if c.get("expected_verdict") == "equivalent"]
    
    for case in equivalent_cases:
        # Create mock candidate match
        mock_candidate = MagicMock()
        mock_candidate.canonical_event = MagicMock()
        
        # Create VerifiedMatch with high cross-encoder score
        verified_match = VerifiedMatch(
            candidate_match=mock_candidate,
            cross_encoder_score=0.85,
            match_type="full_match",
            nli_scores={"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
            primary_event_score=0.9
        )
        
        result = await verifier.verify_pair_async(
            verified_match,
            case["spec_a"],
            case["spec_b"],
            f"market-a-{case['name']}",
            f"market-b-{case['name']}"
        )
        
        assert result.verdict == "equivalent", f"Failed for case: {case['name']}"
        if "expected_confidence_min" in case:
            assert result.confidence >= case["expected_confidence_min"], \
                f"Confidence too low for case: {case['name']}"
        
        if "expected_outcome_mapping" in case:
            # Check outcome mapping
            for key, value in case["expected_outcome_mapping"].items():
                assert result.outcome_mapping.get(key) == value, \
                    f"Outcome mapping mismatch for case: {case['name']}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_golden_dataset_not_equivalent_pairs():
    """Test verification against known non-equivalent pairs."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    golden_dataset = get_golden_dataset()
    not_equivalent_cases = [
        c for c in golden_dataset if c.get("expected_verdict") == "not_equivalent"
    ]
    
    for case in not_equivalent_cases:
        # Create mock candidate match
        mock_candidate = MagicMock()
        mock_candidate.canonical_event = MagicMock()
        
        # Create VerifiedMatch with moderate cross-encoder score
        verified_match = VerifiedMatch(
            candidate_match=mock_candidate,
            cross_encoder_score=0.5,  # Lower score for non-equivalent
            match_type="no_match",
            nli_scores={"entailment": 0.3, "neutral": 0.4, "contradiction": 0.3},
            primary_event_score=0.3
        )
        
        result = await verifier.verify_pair_async(
            verified_match,
            case["spec_a"],
            case["spec_b"],
            f"market-a-{case['name']}",
            f"market-b-{case['name']}"
        )
        
        assert result.verdict == "not_equivalent", f"Failed for case: {case['name']}"
        if "expected_confidence_max" in case:
            assert result.confidence <= case["expected_confidence_max"], \
                f"Confidence too high for case: {case['name']}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_golden_dataset_needs_review_pairs():
    """Test verification against ambiguous pairs."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    golden_dataset = get_golden_dataset()
    needs_review_cases = [
        c for c in golden_dataset if c.get("expected_verdict") == "needs_review"
    ]
    
    for case in needs_review_cases:
        # Create mock candidate match
        mock_candidate = MagicMock()
        mock_candidate.canonical_event = MagicMock()
        
        # Create VerifiedMatch with moderate cross-encoder score
        verified_match = VerifiedMatch(
            candidate_match=mock_candidate,
            cross_encoder_score=0.7,
            match_type="partial_match",
            nli_scores={"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2},
            primary_event_score=0.6
        )
        
        result = await verifier.verify_pair_async(
            verified_match,
            case["spec_a"],
            case["spec_b"],
            f"market-a-{case['name']}",
            f"market-b-{case['name']}"
        )
        
        assert result.verdict == "needs_review", f"Failed for case: {case['name']}"
        if "expected_confidence_min" in case:
            assert result.confidence >= case["expected_confidence_min"], \
                f"Confidence too low for case: {case['name']}"
        if "expected_confidence_max" in case:
            assert result.confidence <= case["expected_confidence_max"], \
                f"Confidence too high for case: {case['name']}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edge_cases():
    """Test edge case handling."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    edge_cases = get_edge_case_dataset()
    
    for case in edge_cases:
        # Create mock candidate match
        mock_candidate = MagicMock()
        mock_candidate.canonical_event = MagicMock()
        
        # Use per-case cross_encoder_score if specified, else default to 0.7
        ce_score = case.get("cross_encoder_score", 0.7)
        verified_match = VerifiedMatch(
            candidate_match=mock_candidate,
            cross_encoder_score=ce_score,
            match_type="partial_match",
            nli_scores={"entailment": 0.6, "neutral": 0.3, "contradiction": 0.1},
            primary_event_score=0.6
        )
        
        result = await verifier.verify_pair_async(
            verified_match,
            case["spec_a"],
            case["spec_b"],
            f"market-a-{case['name']}",
            f"market-b-{case['name']}"
        )
        
        assert result.verdict == case["expected_verdict"], \
            f"Failed for edge case: {case['name']}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_verification_flow(sample_contract_spec_a, sample_contract_spec_b):
    """Test full verification flow with all components."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    # Create mock candidate match
    mock_candidate = MagicMock()
    mock_candidate.canonical_event = MagicMock()
    
    # Create VerifiedMatch
    verified_match = VerifiedMatch(
        candidate_match=mock_candidate,
        cross_encoder_score=0.85,
        match_type="full_match",
        nli_scores={"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
        primary_event_score=0.9,
        secondary_clause_score=0.8
    )
    
    result = await verifier.verify_pair_async(
        verified_match,
        sample_contract_spec_a,
        sample_contract_spec_b,
        "market-a-123",
        "market-b-456"
    )
    
    assert isinstance(result, VerifiedPair)
    assert result.verdict in ["equivalent", "not_equivalent", "needs_review"]
    assert 0.0 <= result.confidence <= 1.0
    assert result.pair_key is not None
    assert result.comparison_details is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_verification_with_missing_fields():
    """Test verification with missing optional fields."""
    verifier = PairVerifier()
    await verifier.initialize()
    
    spec_a = ContractSpec(
        statement="Will Bitcoin reach $100k?",
        resolution_date=None,  # Missing date
        entities=[],  # No entities
        thresholds=[],  # No thresholds
        outcome_labels=["Yes", "No"]
    )
    
    spec_b = ContractSpec(
        statement="Will BTC reach $100k?",
        resolution_date=None,
        entities=[],
        thresholds=[],
        outcome_labels=["Yes", "No"]
    )
    
    # Create mock candidate match
    mock_candidate = MagicMock()
    mock_candidate.canonical_event = MagicMock()
    
    verified_match = VerifiedMatch(
        candidate_match=mock_candidate,
        cross_encoder_score=0.8,
        match_type="full_match",
        nli_scores={"entailment": 0.85, "neutral": 0.1, "contradiction": 0.05},
        primary_event_score=0.85
    )
    
    result = await verifier.verify_pair_async(
        verified_match,
        spec_a,
        spec_b,
        "market-a",
        "market-b"
    )
    
    # Should still produce a result
    assert isinstance(result, VerifiedPair)
    assert result.verdict in ["equivalent", "not_equivalent", "needs_review"]

