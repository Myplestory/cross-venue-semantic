"""
Data types for matching module.

Defines candidate match structures for retrieval phase output.
"""

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.types import CanonicalEvent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from canonicalization.contract_spec import ContractSpec


@dataclass
class CandidateMatch:
    """
    A candidate match from retrieval phase.
    
    Output of retrieval, input to cross-encoder verification.
    """
    canonical_event: CanonicalEvent
    similarity_score: float
    embedding: List[float]
    retrieved_at: datetime = None
    retrieval_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize defaults and validate similarity score."""
        if self.retrieved_at is None:
            self.retrieved_at = datetime.now(UTC)
        
        if self.retrieval_metadata is None:
            self.retrieval_metadata = {}
        
        # Validate and clamp similarity score to [0.0, 1.0]
        # Use epsilon tolerance for floating-point precision issues (e.g., 1.0000001 from Qdrant)
        # but raise ValueError for clearly invalid values (e.g., 1.5, -0.1)
        EPSILON = 1e-6  # Tolerance for floating-point precision errors
        
        if self.similarity_score < -EPSILON or self.similarity_score > 1.0 + EPSILON:
            raise ValueError(
                f"Similarity score must be between 0.0 and 1.0, got {self.similarity_score}"
            )
        
        # Clamp to [0.0, 1.0] to handle floating-point precision within epsilon
        # Qdrant may return scores slightly above 1.0 (e.g., 1.0000001) due to cosine similarity
        # calculations with floating-point arithmetic
        self.similarity_score = max(0.0, min(1.0, self.similarity_score))
        
        self.embedding = list(self.embedding)


@dataclass
class VerifiedMatch:
    """
    A verified match after cross-encoder re-ranking.
    
    Output of cross-encoder phase, input to LLM verifier (optional).
    """
    candidate_match: CandidateMatch  # Original candidate from retrieval
    cross_encoder_score: float  # Combined confidence (0-1)
    match_type: str  # "full_match", "partial_match", "no_match"
    nli_scores: Dict[str, Any]  # Raw entailment/neutral/contradiction scores
    primary_event_score: float  # Primary event equivalence score
    secondary_clause_score: Optional[float] = None  # Secondary clause equivalence score
    verified_at: datetime = None
    verification_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize defaults and validate scores."""
        if self.verified_at is None:
            self.verified_at = datetime.now(UTC)
        
        if self.verification_metadata is None:
            self.verification_metadata = {}
        
        # Validate confidence score
        EPSILON = 1e-6
        if self.cross_encoder_score < -EPSILON or self.cross_encoder_score > 1.0 + EPSILON:
            raise ValueError(
                f"Cross-encoder score must be between 0.0 and 1.0, got {self.cross_encoder_score}"
            )
        self.cross_encoder_score = max(0.0, min(1.0, self.cross_encoder_score))
        
        # Validate match type
        if self.match_type not in ["full_match", "partial_match", "no_match"]:
            raise ValueError(f"Invalid match_type: {self.match_type}")
        
        # Validate primary event score
        if not (0.0 <= self.primary_event_score <= 1.0):
            self.primary_event_score = max(0.0, min(1.0, self.primary_event_score))
        
        # Validate secondary clause score if provided
        if self.secondary_clause_score is not None:
            if not (0.0 <= self.secondary_clause_score <= 1.0):
                self.secondary_clause_score = max(0.0, min(1.0, self.secondary_clause_score))


@dataclass
class VerifiedPair:
    """
    Verified pair after ContractSpec comparison.
    
    Output of pair verification, input to persistence (Phase 8).
    """
    pair_key: str  # Sorted market IDs (stable identifier)
    market_a_id: str
    market_b_id: str
    contract_spec_a: 'ContractSpec'  # Forward reference
    contract_spec_b: 'ContractSpec'  # Forward reference
    outcome_mapping: Dict[str, str]  # {"YES_A": "YES_B", "NO_A": "NO_B"}
    verdict: str  # "equivalent", "not_equivalent", "needs_review"
    confidence: float  # Overall confidence (0-1)
    comparison_details: Dict[str, Any]  # Detailed comparison results
    verified_at: datetime = None
    
    def __post_init__(self):
        """Initialize defaults and validate."""
        if self.verified_at is None:
            self.verified_at = datetime.now(UTC)
        
        if self.verdict not in ["equivalent", "not_equivalent", "needs_review"]:
            raise ValueError(f"Invalid verdict: {self.verdict}")
        
        # Validate confidence
        EPSILON = 1e-6
        if self.confidence < -EPSILON or self.confidence > 1.0 + EPSILON:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        self.confidence = max(0.0, min(1.0, self.confidence))

