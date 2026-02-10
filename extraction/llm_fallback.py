"""
LLM-based extraction for complex cases.

Uses GPT-4 or Claude with structured outputs (JSON mode).
Protected by circuit breaker to prevent cascade failures.
"""

import logging
from typing import List, Optional

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.contract_spec import ContractSpec
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitBreakerConfig
import config

logger = logging.getLogger(__name__)


class LLMFallback:
    """
    LLM-based extraction with circuit breaker protection.
    
    Uses GPT-4 or Claude with structured outputs (JSON mode).
    Only called when rule-based confidence is low.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        self.model = model or config.EXTRACTION_LLM_MODEL
        self.api_key = api_key or config.EXTRACTION_LLM_API_KEY
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )
        self._client = None
        
        if HAS_OPENAI and self.api_key:
            self._client = AsyncOpenAI(api_key=self.api_key)
    
    async def extract_with_llm(
        self,
        canonical_text: str,
        failed_fields: List[str]
    ) -> ContractSpec:
        """
        Extract ContractSpec using LLM with circuit breaker protection.
        
        Args:
            canonical_text: Full canonical text
            failed_fields: Fields that failed rule-based extraction
            
        Returns:
            ContractSpec from LLM
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            RuntimeError: If LLM call fails
        """
        if self.circuit_breaker.is_open():
            raise CircuitBreakerOpenError(
                "LLM fallback disabled: Circuit breaker is OPEN. "
                "Service may be down or rate-limited."
            )
        
        if not HAS_OPENAI or not self._client:
            raise RuntimeError("OpenAI client not available")
        
        return await self.circuit_breaker.call(
            self._call_llm_api,
            canonical_text,
            failed_fields
        )
    
    async def _call_llm_api(
        self,
        canonical_text: str,
        failed_fields: List[str]
    ) -> ContractSpec:
        """Actual LLM API call (protected by circuit breaker)."""
        prompt = self._build_prompt(canonical_text, failed_fields)
        
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a contract specification extractor. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        
        json_str = response.choices[0].message.content
        data = ContractSpec.model_validate_json(json_str)
        
        data.confidence = 0.9
        data.extraction_notes = f"LLM extraction using {self.model}"
        
        return data
    
    def _build_prompt(
        self,
        canonical_text: str,
        failed_fields: List[str]
    ) -> str:
        """Build prompt for LLM extraction."""
        schema = ContractSpec.model_json_schema()
        
        return f"""Extract structured contract specifications from the following market text:

{canonical_text}

Focus on extracting these fields (especially: {', '.join(failed_fields) if failed_fields else 'all fields'}):
- statement: Core market question
- dates: Resolution date and event date (if applicable)
- entities: People, organizations, locations
- thresholds: Numeric values with comparison operators
- resolution_criteria: Detailed resolution rules
- data_source: Where resolution data comes from
- outcome_labels: Available outcomes

Return a JSON object matching this schema:
{str(schema)}

Ensure all dates are in ISO 8601 format."""

