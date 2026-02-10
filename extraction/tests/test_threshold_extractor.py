"""Unit tests for ThresholdExtractor."""

import pytest
from extraction.parsers.threshold_extractor import ThresholdExtractor


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_thresholds_currency(sample_canonical_text):
    """Test extracting currency thresholds."""
    extractor = ThresholdExtractor()
    thresholds = await extractor.extract_thresholds(
        sample_canonical_text.split('\n')[1],
        None
    )
    
    assert len(thresholds) > 0
    assert any(t.value == 60000.0 and t.unit == "dollars" for t in thresholds)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_thresholds_percentage():
    """Test extracting percentage thresholds."""
    extractor = ThresholdExtractor()
    thresholds = await extractor.extract_thresholds(
        "Will the price be above 50%?",
        None
    )
    
    assert len(thresholds) > 0
    assert any(t.value == 50.0 and t.unit == "percentage" for t in thresholds)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_thresholds_with_negation(sample_canonical_text_with_negation):
    """Test extracting thresholds with negation."""
    extractor = ThresholdExtractor()
    statement = sample_canonical_text_with_negation.split('\n')[1]
    criteria = sample_canonical_text_with_negation.split('Resolution Criteria:')[1].split('\n')[0]
    
    thresholds = await extractor.extract_thresholds(statement, criteria)
    
    assert len(thresholds) > 0
    assert any(t.is_negated is True for t in thresholds)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_thresholds_multiple(sample_canonical_text_multi_threshold):
    """Test extracting multiple thresholds."""
    extractor = ThresholdExtractor()
    statement = sample_canonical_text_multi_threshold.split('\n')[1]
    criteria = sample_canonical_text_multi_threshold.split('Resolution Criteria:')[1].split('\n')[0]
    
    thresholds = await extractor.extract_thresholds(statement, criteria)
    
    assert len(thresholds) >= 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_thresholds_comparison_operators():
    """Test extracting different comparison operators."""
    extractor = ThresholdExtractor()
    
    test_cases = [
        ("above $60k", ">"),
        ("below $50k", "<"),
        ("at least $60k", ">="),
        ("at most $80k", "<="),
        ("exactly $60k", "=="),
    ]
    
    for text, expected_op in test_cases:
        thresholds = await extractor.extract_thresholds(text, None)
        if thresholds:
            assert thresholds[0].comparison == expected_op


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_negation():
    """Test negation detection."""
    extractor = ThresholdExtractor()
    
    text = "Will Bitcoin not exceed $100k by end of year?"
    is_negated, context = extractor._detect_negation(text, text.find("$100k"))
    
    assert is_negated is True
    assert context is not None

