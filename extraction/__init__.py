"""
ContractSpec extraction module.

Extracts structured ContractSpec from canonical text using rule-based
parsing with LLM fallback for complex cases.
"""

from .spec_extractor import ContractSpecExtractor
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitBreakerConfig
from .llm_fallback import LLMFallback

__all__ = [
    "ContractSpecExtractor",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitBreakerConfig",
    "LLMFallback",
]
