"""Unit tests for SectionParser."""

import pytest
from extraction.parsers.section_parser import SectionParser


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_statement(sample_canonical_text):
    """Test parsing Market Statement section."""
    parser = SectionParser()
    statement, span = await parser.parse_statement(sample_canonical_text)
    
    assert "Bitcoin" in statement
    assert "$60,000" in statement
    assert "Coinbase" in statement
    assert span is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_statement_missing_section():
    """Test parsing when Market Statement section is missing."""
    text = "Just some text without sections"
    parser = SectionParser()
    statement, span = await parser.parse_statement(text)
    
    assert statement == "Just some text without sections"
    assert span is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_resolution_criteria(sample_canonical_text):
    """Test parsing Resolution Criteria section."""
    parser = SectionParser()
    criteria, span = await parser.parse_resolution_criteria(sample_canonical_text)
    
    assert criteria is not None
    assert "Coinbase" in criteria
    assert "BTC/USD" in criteria
    assert span is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_resolution_criteria_missing():
    """Test parsing when Resolution Criteria section is missing."""
    text = "Market Statement:\nTest\nEnd Date: 2024-12-31"
    parser = SectionParser()
    criteria, span = await parser.parse_resolution_criteria(text)
    
    assert criteria is None
    assert span is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_outcomes(sample_canonical_text):
    """Test parsing Outcomes section."""
    parser = SectionParser()
    outcomes, spans = await parser.parse_outcomes(sample_canonical_text)
    
    assert len(outcomes) == 2
    assert "Yes" in outcomes
    assert "No" in outcomes
    assert spans is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_outcomes_missing():
    """Test parsing when Outcomes section is missing."""
    text = "Market Statement:\nTest"
    parser = SectionParser()
    outcomes, spans = await parser.parse_outcomes(text)
    
    assert outcomes == ["Yes", "No"]
    assert spans is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_end_date(sample_canonical_text):
    """Test parsing End Date section."""
    parser = SectionParser()
    date_str, span = await parser.parse_end_date(sample_canonical_text)
    
    assert date_str == "2024-12-31"
    assert span is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_end_date_missing():
    """Test parsing when End Date section is missing."""
    text = "Market Statement:\nTest"
    parser = SectionParser()
    date_str, span = await parser.parse_end_date(text)
    
    assert date_str is None
    assert span is None

