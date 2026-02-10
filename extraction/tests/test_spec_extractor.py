"""Unit tests for ContractSpecExtractor."""

import pytest
import hashlib
from unittest.mock import AsyncMock, patch
from extraction.spec_extractor import ContractSpecExtractor
from extraction.circuit_breaker import CircuitBreakerOpenError
from canonicalization.contract_spec import ContractSpec


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extractor_initialization():
    """Test extractor initialization."""
    extractor = ContractSpecExtractor()
    await extractor.initialize()
    
    assert extractor._initialized is True
    assert extractor.section_parser is not None
    assert extractor.date_parser is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_rule_based(sample_canonical_text):
    """Test rule-based extraction."""
    extractor = ContractSpecExtractor(use_llm_fallback=False)
    await extractor.initialize()
    
    spec, confidence = await extractor._extract_rule_based(sample_canonical_text)
    
    assert isinstance(spec, ContractSpec)
    assert spec.statement is not None
    assert confidence >= 0.0
    assert confidence <= 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_with_cache_hit(sample_canonical_text, sample_contract_spec):
    """Test extraction with cache hit."""
    extractor = ContractSpecExtractor(use_llm_fallback=False)
    await extractor.initialize()
    
    content_hash = hashlib.sha256(sample_canonical_text.encode()).hexdigest()
    await extractor.cache.set(content_hash, sample_contract_spec)
    
    spec = await extractor.extract_async(sample_canonical_text, content_hash)
    
    assert spec == sample_contract_spec


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_high_confidence_early_exit(sample_canonical_text):
    """Test early exit for high confidence extraction."""
    extractor = ContractSpecExtractor(
        use_llm_fallback=True,
        high_confidence_threshold=0.9
    )
    await extractor.initialize()
    
    with patch.object(extractor, '_extract_rule_based', new_callable=AsyncMock) as mock_extract:
        mock_spec = ContractSpec(statement="Test", confidence=0.95)
        mock_extract.return_value = (mock_spec, 0.95)
        
        spec = await extractor.extract_async(sample_canonical_text)
        
        assert spec.confidence == 0.95
        assert extractor.llm_fallback is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_llm_fallback_low_confidence(sample_canonical_text):
    """Test LLM fallback for low confidence."""
    extractor = ContractSpecExtractor(
        use_llm_fallback=True,
        confidence_threshold=0.7
    )
    await extractor.initialize()
    
    with patch.object(extractor, '_extract_rule_based', new_callable=AsyncMock) as mock_extract:
        with patch.object(extractor.llm_fallback, 'extract_with_llm', new_callable=AsyncMock) as mock_llm:
            mock_spec = ContractSpec(statement="Test", confidence=0.5)
            mock_extract.return_value = (mock_spec, 0.5)
            
            mock_llm_spec = ContractSpec(statement="LLM extracted", confidence=0.9)
            mock_llm.return_value = mock_llm_spec
            
            spec = await extractor.extract_async(sample_canonical_text)
            
            assert spec.confidence == 0.9
            mock_llm.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_llm_fallback_circuit_breaker_open(sample_canonical_text):
    """Test LLM fallback handles circuit breaker open."""
    extractor = ContractSpecExtractor(
        use_llm_fallback=True,
        confidence_threshold=0.7
    )
    await extractor.initialize()
    
    with patch.object(extractor, '_extract_rule_based', new_callable=AsyncMock) as mock_extract:
        with patch.object(extractor.llm_fallback, 'extract_with_llm', new_callable=AsyncMock) as mock_llm:
            mock_spec = ContractSpec(statement="Test", confidence=0.5)
            mock_extract.return_value = (mock_spec, 0.5)
            
            mock_llm.side_effect = CircuitBreakerOpenError("Circuit open")
            
            spec = await extractor.extract_async(sample_canonical_text)
            
            assert spec.confidence == 0.5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_calculate_confidence():
    """Test confidence calculation."""
    extractor = ContractSpecExtractor()
    
    spec = ContractSpec(
        statement="Test statement",
        resolution_criteria="Test criteria",
        outcome_labels=["Yes", "No"],
        resolution_date=None,
        entities=[],
        thresholds=[],
        data_source=None
    )
    
    confidence = extractor._calculate_confidence(spec)
    
    assert confidence >= 0.0
    assert confidence <= 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_failed_fields():
    """Test getting failed fields."""
    extractor = ContractSpecExtractor()
    
    spec = ContractSpec(
        statement="",
        resolution_date=None,
        entities=[],
        thresholds=[],
        data_source=None
    )
    
    failed = extractor._get_failed_fields(spec, 0.3)
    
    assert "statement" in failed
    assert "resolution_date" in failed
    assert "entities" in failed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evidence_spans_tracking(sample_canonical_text):
    """Test evidence spans tracking when enabled."""
    extractor = ContractSpecExtractor(track_evidence_spans=True)
    await extractor.initialize()
    
    spec, confidence = await extractor._extract_rule_based(sample_canonical_text)
    
    assert spec.evidence_spans is not None
    assert len(spec.evidence_spans) > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evidence_spans_disabled(sample_canonical_text):
    """Test evidence spans not tracked when disabled."""
    extractor = ContractSpecExtractor(track_evidence_spans=False)
    await extractor.initialize()
    
    spec, confidence = await extractor._extract_rule_based(sample_canonical_text)
    
    assert spec.evidence_spans == {}

