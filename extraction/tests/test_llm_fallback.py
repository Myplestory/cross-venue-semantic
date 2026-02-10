"""Unit tests for LLMFallback."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from extraction.llm_fallback import LLMFallback
from extraction.circuit_breaker import CircuitBreakerOpenError
from canonicalization.contract_spec import ContractSpec


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_fallback_circuit_breaker_open():
    """Test LLM fallback raises error when circuit breaker is open."""
    from extraction.circuit_breaker import CircuitState
    
    fallback = LLMFallback()
    fallback.circuit_breaker.state = CircuitState.OPEN
    
    with pytest.raises(CircuitBreakerOpenError):
        await fallback.extract_with_llm("test text", [])


@pytest.mark.unit
@pytest.mark.asyncio
@patch('extraction.llm_fallback.HAS_OPENAI', True)
async def test_llm_fallback_extraction(mock_openai_client):
    """Test LLM fallback extraction."""
    fallback = LLMFallback()
    fallback._client = mock_openai_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"statement": "Test statement", "confidence": 0.9}'
    
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    spec = await fallback.extract_with_llm("test text", [])
    
    assert isinstance(spec, ContractSpec)
    assert spec.confidence == 0.9


@pytest.mark.unit
@pytest.mark.asyncio
@patch('extraction.llm_fallback.HAS_OPENAI', False)
async def test_llm_fallback_no_openai():
    """Test LLM fallback raises error when OpenAI is not available."""
    fallback = LLMFallback()
    
    with pytest.raises(RuntimeError, match="OpenAI client not available"):
        await fallback.extract_with_llm("test text", [])


@pytest.mark.unit
@pytest.mark.asyncio
@patch('extraction.llm_fallback.HAS_OPENAI', True)
async def test_llm_fallback_api_error(mock_openai_client):
    """Test LLM fallback handles API errors."""
    fallback = LLMFallback()
    fallback._client = mock_openai_client
    
    mock_openai_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
    
    with pytest.raises(Exception, match="API error"):
        await fallback.extract_with_llm("test text", [])

