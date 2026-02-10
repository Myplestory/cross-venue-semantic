"""
Matching Module

Two-stage matching pipeline:
1. Embedding-based candidate retrieval (high recall)
2. Cross-encoder verification (high precision)
3. Pair verification (ContractSpec comparison)
"""

from .types import CandidateMatch, VerifiedMatch, VerifiedPair
from .retriever import CandidateRetriever
from .cross_encoder import CrossEncoder
from .reranker import CandidateReranker
from .pair_verifier import PairVerifier

__all__ = [
    "CandidateMatch",
    "VerifiedMatch",
    "VerifiedPair",
    "CandidateRetriever",
    "CrossEncoder",
    "CandidateReranker",
    "PairVerifier",
]
