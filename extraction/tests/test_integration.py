"""Integration tests for ContractSpec extraction."""

import pytest
from extraction.spec_extractor import ContractSpecExtractor
from canonicalization.contract_spec import ContractSpec


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_extraction_pipeline(sample_canonical_text):
    """Test full extraction pipeline end-to-end."""
    extractor = ContractSpecExtractor(use_llm_fallback=False)
    await extractor.initialize()
    
    spec = await extractor.extract_async(sample_canonical_text)
    
    assert isinstance(spec, ContractSpec)
    assert spec.statement is not None
    assert len(spec.statement) > 0
    assert spec.confidence > 0.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_with_all_fields(sample_canonical_text):
    """Test extraction extracts all available fields."""
    extractor = ContractSpecExtractor(use_llm_fallback=False)
    await extractor.initialize()
    
    spec = await extractor.extract_async(sample_canonical_text)
    
    assert spec.statement is not None
    assert spec.resolution_date is not None
    assert spec.outcome_labels is not None
    assert len(spec.outcome_labels) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_caching(sample_canonical_text):
    """Test extraction caching works correctly."""
    extractor = ContractSpecExtractor(use_llm_fallback=False)
    await extractor.initialize()
    
    spec1 = await extractor.extract_async(sample_canonical_text)
    spec2 = await extractor.extract_async(sample_canonical_text)
    
    assert spec1.statement == spec2.statement
    assert spec1.confidence == spec2.confidence


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_parallel_fields(sample_canonical_text):
    """Test parallel field extraction works."""
    extractor = ContractSpecExtractor(use_llm_fallback=False)
    await extractor.initialize()
    
    spec, confidence = await extractor._extract_rule_based(sample_canonical_text)
    
    assert spec.statement is not None
    assert spec.resolution_criteria is not None
    assert spec.outcome_labels is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_simple_text(sample_canonical_text_simple):
    """Test extraction with minimal canonical text."""
    extractor = ContractSpecExtractor(use_llm_fallback=False)
    await extractor.initialize()
    
    spec = await extractor.extract_async(sample_canonical_text_simple)
    
    assert isinstance(spec, ContractSpec)
    assert spec.statement is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_with_negation(sample_canonical_text_with_negation):
    """Test extraction handles negation patterns."""
    extractor = ContractSpecExtractor(use_llm_fallback=False)
    await extractor.initialize()
    
    spec = await extractor.extract_async(sample_canonical_text_with_negation)
    
    assert isinstance(spec, ContractSpec)
    if spec.thresholds:
        assert any(t.is_negated for t in spec.thresholds)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_performance(sample_canonical_text):
    """Test extraction performance (should be <200ms for rule-based)."""
    import time
    
    extractor = ContractSpecExtractor(use_llm_fallback=False)
    await extractor.initialize()
    
    start = time.time()
    spec = await extractor.extract_async(sample_canonical_text)
    elapsed = time.time() - start
    
    assert elapsed < 0.2
    assert isinstance(spec, ContractSpec)

