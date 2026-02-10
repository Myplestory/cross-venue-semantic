"""Unit tests for DateParser."""

import pytest
from datetime import datetime
from extraction.parsers.date_parser import DateParser


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_dates_iso_format(sample_canonical_text):
    """Test parsing ISO format date (fast path)."""
    parser = DateParser()
    resolution_date, event_date = await parser.parse_dates(sample_canonical_text, "2024-12-31")
    
    assert resolution_date is not None
    assert resolution_date.date == datetime(2024, 12, 31)
    assert resolution_date.is_deadline is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_dates_complex_format():
    """Test parsing complex date format (fallback to dateutil)."""
    from extraction.parsers.date_parser import HAS_DATEUTIL
    
    parser = DateParser()
    resolution_date, event_date = await parser.parse_dates(
        "Test",
        "December 31, 2024"
    )
    
    if HAS_DATEUTIL:
        assert resolution_date is not None
        assert resolution_date.is_deadline is True
    else:
        pytest.skip("python-dateutil not available")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_dates_invalid_format():
    """Test parsing invalid date format."""
    parser = DateParser()
    resolution_date, event_date = await parser.parse_dates("Test", "invalid-date")
    
    assert resolution_date is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_dates_no_end_date():
    """Test parsing when no end date provided."""
    parser = DateParser()
    resolution_date, event_date = await parser.parse_dates("Test", None)
    
    assert resolution_date is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_event_date():
    """Test extracting event date from statement context."""
    text = """Market Statement:
Will Bitcoin close on Dec 31, 2024?

End Date: 2024-12-31"""
    
    parser = DateParser()
    resolution_date, event_date = await parser.parse_dates(text, "2024-12-31")
    
    assert resolution_date is not None
    assert event_date is None

