"""Unit tests for ThresholdComparator."""

import pytest

from matching.comparators.threshold_comparator import ThresholdComparator
from canonicalization.contract_spec import ThresholdSpec


@pytest.mark.unit
@pytest.mark.asyncio
async def test_threshold_comparator_initialization():
    """Test ThresholdComparator initialization."""
    comparator = ThresholdComparator()
    await comparator.initialize()
    
    assert comparator._initialized is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_thresholds_both_empty():
    """Test comparing empty threshold lists."""
    comparator = ThresholdComparator()
    await comparator.initialize()
    
    score, details = await comparator.compare_thresholds([], [])
    
    assert score == 1.0
    assert details["match"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_thresholds_one_empty():
    """Test comparing when one list is empty."""
    comparator = ThresholdComparator()
    await comparator.initialize()
    
    thresholds_a = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
    ]
    
    score, details = await comparator.compare_thresholds(thresholds_a, [])
    
    assert score == 0.0
    assert details["match"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_thresholds_exact_match():
    """Test exact threshold matching."""
    comparator = ThresholdComparator()
    await comparator.initialize()
    
    thresholds_a = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
    ]
    thresholds_b = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
    ]
    
    score, details = await comparator.compare_thresholds(thresholds_a, thresholds_b, tolerance_percent=0.01)
    
    assert score == 1.0
    assert details["match"] is True
    assert details["matched_count"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_thresholds_within_tolerance():
    """Test threshold matching within tolerance."""
    comparator = ThresholdComparator()
    await comparator.initialize()
    
    thresholds_a = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
    ]
    thresholds_b = [
        ThresholdSpec(value=60050.0, unit="dollars", comparison=">")  # 0.08% difference
    ]
    
    score, details = await comparator.compare_thresholds(thresholds_a, thresholds_b, tolerance_percent=0.01)
    
    # Should match within 1% tolerance
    assert score == 1.0
    assert details["match"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_thresholds_outside_tolerance():
    """Test threshold matching outside tolerance."""
    comparator = ThresholdComparator()
    await comparator.initialize()
    
    thresholds_a = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
    ]
    thresholds_b = [
        ThresholdSpec(value=70000.0, unit="dollars", comparison=">")  # 16.7% difference
    ]
    
    score, details = await comparator.compare_thresholds(thresholds_a, thresholds_b, tolerance_percent=0.01)
    
    assert score == 0.0
    assert details["match"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_thresholds_different_comparison():
    """Test threshold matching with different comparison operators."""
    comparator = ThresholdComparator()
    await comparator.initialize()
    
    thresholds_a = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">")
    ]
    thresholds_b = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">=")
    ]
    
    score, details = await comparator.compare_thresholds(thresholds_a, thresholds_b)
    
    # Different comparison operators should not match
    assert score == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_thresholds_negation_mismatch():
    """Test threshold matching with negation mismatch."""
    comparator = ThresholdComparator()
    await comparator.initialize()
    
    thresholds_a = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">", is_negated=False)
    ]
    thresholds_b = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">", is_negated=True)
    ]
    
    score, details = await comparator.compare_thresholds(thresholds_a, thresholds_b)
    
    # Negation mismatch should not match
    assert score == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_thresholds_multiple():
    """Test comparing multiple thresholds."""
    comparator = ThresholdComparator()
    await comparator.initialize()
    
    thresholds_a = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">"),
        ThresholdSpec(value=80000.0, unit="dollars", comparison="<")
    ]
    thresholds_b = [
        ThresholdSpec(value=60000.0, unit="dollars", comparison=">"),
        ThresholdSpec(value=80000.0, unit="dollars", comparison="<")
    ]
    
    score, details = await comparator.compare_thresholds(thresholds_a, thresholds_b)
    
    assert score == 1.0
    assert details["matched_count"] == 2

