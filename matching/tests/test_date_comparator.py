"""Unit tests for DateComparator."""

import pytest
from datetime import datetime, timedelta

from matching.comparators.date_comparator import DateComparator
from canonicalization.contract_spec import DateSpec


@pytest.mark.unit
@pytest.mark.asyncio
async def test_date_comparator_initialization():
    """Test DateComparator initialization."""
    comparator = DateComparator()
    await comparator.initialize()
    
    assert comparator._initialized is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_dates_both_missing():
    """Test comparing when both dates are missing."""
    comparator = DateComparator()
    await comparator.initialize()
    
    score, details = await comparator.compare_dates(None, None)
    
    assert score == 0.5
    assert details["match"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_dates_one_missing():
    """Test comparing when one date is missing."""
    comparator = DateComparator()
    await comparator.initialize()
    
    date_a = DateSpec(
        date=datetime(2024, 12, 31),
        is_deadline=True
    )
    
    score, details = await comparator.compare_dates(date_a, None)
    
    assert score == 0.0
    assert details["match"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_dates_exact_match():
    """Test exact date matching."""
    comparator = DateComparator()
    await comparator.initialize()
    
    date_a = DateSpec(
        date=datetime(2024, 12, 31),
        is_deadline=True
    )
    date_b = DateSpec(
        date=datetime(2024, 12, 31),
        is_deadline=True
    )
    
    score, details = await comparator.compare_dates(date_a, date_b, tolerance_days=1)
    
    assert score == 1.0
    assert details["match"] is True
    assert details["date_diff_days"] == 0
    assert details["deadline_match"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_dates_within_tolerance():
    """Test date matching within tolerance."""
    comparator = DateComparator()
    await comparator.initialize()
    
    date_a = DateSpec(
        date=datetime(2024, 12, 31),
        is_deadline=True
    )
    date_b = DateSpec(
        date=datetime(2025, 1, 1),  # 1 day difference
        is_deadline=True
    )
    
    score, details = await comparator.compare_dates(date_a, date_b, tolerance_days=1)
    
    assert score == 0.9
    assert details["match"] is True
    assert details["date_diff_days"] == 1
    assert details["deadline_match"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_dates_outside_tolerance():
    """Test date matching outside tolerance."""
    comparator = DateComparator()
    await comparator.initialize()
    
    date_a = DateSpec(
        date=datetime(2024, 12, 31),
        is_deadline=True
    )
    date_b = DateSpec(
        date=datetime(2025, 1, 15),  # 15 days difference
        is_deadline=True
    )
    
    score, details = await comparator.compare_dates(date_a, date_b, tolerance_days=1)
    
    assert score == 0.0
    assert details["match"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_dates_deadline_mismatch():
    """Test date matching with deadline flag mismatch."""
    comparator = DateComparator()
    await comparator.initialize()
    
    date_a = DateSpec(
        date=datetime(2024, 12, 31),
        is_deadline=True
    )
    date_b = DateSpec(
        date=datetime(2024, 12, 31),
        is_deadline=False  # Different flag
    )
    
    score, details = await comparator.compare_dates(date_a, date_b, tolerance_days=1)
    
    # Should match on date but penalize for deadline mismatch
    assert score <= 0.5
    assert details["deadline_match"] is False

