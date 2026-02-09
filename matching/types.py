"""
Data types for matching module.

Defines candidate match structures for retrieval phase output.
"""

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.types import CanonicalEvent


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

