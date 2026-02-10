"""Unit tests for EntityExtractor."""

import pytest
from unittest.mock import patch, MagicMock
from extraction.parsers.entity_extractor import EntityExtractor


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_entities_simple_case():
    """Test entity extraction skips for simple text."""
    extractor = EntityExtractor()
    await extractor.initialize()
    
    entities = await extractor.extract_entities("Short text", None)
    
    assert entities == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_entities_no_capitalization():
    """Test entity extraction skips text without capitalization."""
    extractor = EntityExtractor()
    await extractor.initialize()
    
    text = "this is a long text without any proper nouns or capitalized words"
    entities = await extractor.extract_entities(text, None)
    
    assert entities == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_entities_with_spacy(mock_spacy_model):
    """Test entity extraction with spaCy."""
    from extraction.parsers.entity_extractor import EntityExtractor
    from unittest.mock import patch
    
    extractor = EntityExtractor()
    
    with patch('extraction.parsers.entity_extractor.HAS_SPACY', True):
        extractor._nlp = mock_spacy_model
        extractor._initialized = True
        
        text = "Bitcoin and Coinbase are mentioned here with proper capitalization"
        entities = await extractor.extract_entities(text, None)
        
        assert len(entities) > 0
        assert any(e.name == "Bitcoin" for e in entities)
        assert any(e.name == "Coinbase" for e in entities)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_entities_no_spacy():
    """Test entity extraction when spaCy is not available."""
    extractor = EntityExtractor()
    extractor._nlp = None
    
    with patch('extraction.parsers.entity_extractor.HAS_SPACY', False):
        entities = await extractor.extract_entities("Bitcoin test", None)
        assert entities == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_has_potential_entities():
    """Test heuristic for potential entities."""
    extractor = EntityExtractor()
    
    assert extractor._has_potential_entities("Bitcoin and Coinbase are here") is True
    assert extractor._has_potential_entities("no proper nouns here") is False
    assert extractor._has_potential_entities("Short") is False

