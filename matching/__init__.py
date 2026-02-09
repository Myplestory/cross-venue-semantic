"""
Matching Module

Two-stage matching pipeline:
1. Embedding-based candidate retrieval (high recall)
2. Cross-encoder verification (high precision)
3. LLM verification (final gate)
"""

from .types import CandidateMatch
from .retriever import CandidateRetriever

__all__ = [
    "CandidateMatch",
    "CandidateRetriever",
]
