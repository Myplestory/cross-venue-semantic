"""
Matching Module

Two-stage matching pipeline:
1. Embedding-based candidate retrieval (high recall)
2. Cross-encoder verification (high precision)
3. LLM verification (final gate)
"""

from .types import CandidateMatch, VerifiedMatch
from .retriever import CandidateRetriever
from .cross_encoder import CrossEncoder
from .reranker import CandidateReranker

__all__ = [
    "CandidateMatch",
    "VerifiedMatch",
    "CandidateRetriever",
    "CrossEncoder",
    "CandidateReranker",
]
