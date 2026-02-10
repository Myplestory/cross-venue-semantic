"""Unit tests for DataSourceExtractor."""

import pytest
from extraction.parsers.data_source_extractor import DataSourceExtractor


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_data_source_coinbase(sample_canonical_text):
    """Test extracting Coinbase as data source."""
    extractor = DataSourceExtractor()
    criteria = sample_canonical_text.split('Resolution Criteria:')[1].split('Clarifications:')[0]
    
    data_source = await extractor.extract_data_source(criteria)
    
    assert data_source == "Coinbase"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_data_source_binance():
    """Test extracting Binance as data source."""
    extractor = DataSourceExtractor()
    criteria = "Market resolves based on Binance BTC/USD price"
    
    data_source = await extractor.extract_data_source(criteria)
    
    assert data_source == "Binance"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_data_source_official_results():
    """Test extracting official results as data source."""
    extractor = DataSourceExtractor()
    criteria = "Market resolves based on official results from election"
    
    data_source = await extractor.extract_data_source(criteria)
    
    assert data_source == "official results"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_data_source_context_pattern():
    """Test extracting data source from context patterns."""
    extractor = DataSourceExtractor()
    criteria = "According to Coinbase, the price will be used"
    
    data_source = await extractor.extract_data_source(criteria)
    
    assert data_source == "Coinbase"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_data_source_none():
    """Test when no data source is found."""
    extractor = DataSourceExtractor()
    criteria = "Market resolves based on general market conditions"
    
    data_source = await extractor.extract_data_source(criteria)
    
    assert data_source is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_data_source_empty():
    """Test with empty resolution criteria."""
    extractor = DataSourceExtractor()
    
    data_source = await extractor.extract_data_source(None)
    
    assert data_source is None

