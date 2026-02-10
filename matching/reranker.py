"""
Candidate re-ranker using cross-encoder for semantic equivalence.

Re-ranks candidates from retrieval phase using cross-encoder scoring.
Evaluates primary event matching and secondary clause equivalence.
"""

import asyncio
import logging
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.types import CanonicalEvent
from .types import CandidateMatch, VerifiedMatch
from .cross_encoder import CrossEncoder

logger = logging.getLogger(__name__)


class CandidateReranker:
    """
    Re-ranks candidates using cross-encoder for semantic equivalence.
    
    Handles:
    - Primary event matching
    - Secondary clause evaluation
    - Confidence score calculation
    - Top-K filtering after re-ranking
    """
    
    def __init__(
        self,
        cross_encoder: CrossEncoder,
        top_k: int = 10,
        score_threshold: float = 0.7,
        primary_weight: float = 0.7,
        secondary_weight: float = 0.3,
    ):
        """
        Initialize candidate reranker.
        
        Args:
            cross_encoder: CrossEncoder instance for scoring
            top_k: Number of top candidates to return after re-ranking
            score_threshold: Minimum combined confidence score (0-1)
            primary_weight: Weight for primary event score in combination
            secondary_weight: Weight for secondary clause score in combination
        """
        self.cross_encoder = cross_encoder
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight
        
        # Validate weights sum to 1.0
        total_weight = primary_weight + secondary_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Primary and secondary weights must sum to 1.0, got {total_weight}"
            )
        
        logger.info(
            f"CandidateReranker initialized: top_k={top_k}, "
            f"score_threshold={score_threshold}, "
            f"primary_weight={primary_weight}, secondary_weight={secondary_weight}"
        )
    
    async def initialize(self) -> None:
        """Initialize cross-encoder model."""
        await self.cross_encoder.initialize()
        logger.debug("CandidateReranker initialized")
    
    async def rerank_async(
        self,
        query_event: CanonicalEvent,
        candidates: List[CandidateMatch],
    ) -> List[VerifiedMatch]:
        """
        Re-rank candidates using cross-encoder.
        
        Args:
            query_event: Query market canonical event
            candidates: List of candidate matches from retrieval
            
        Returns:
            List of verified matches, sorted by confidence (highest first)
        """
        if not candidates:
            return []
        
        # Extract primary event and secondary clauses from query
        query_text = query_event.canonical_text
        query_primary = self.cross_encoder.extract_primary_event(query_text)
        query_clauses = self.cross_encoder.extract_secondary_clauses(query_text)
        
        # Score primary events for all candidates
        candidate_texts = [c.canonical_event.canonical_text for c in candidates]
        candidate_primaries = [
            self.cross_encoder.extract_primary_event(text) for text in candidate_texts
        ]
        
        # Batch score primary events
        primary_pairs = [(query_primary, cp) for cp in candidate_primaries]
        primary_nli_scores = await self.cross_encoder.score_batch_async(primary_pairs)
        
        # Process each candidate
        verified_matches = []
        for i, candidate in enumerate(candidates):
            primary_nli = primary_nli_scores[i]
            primary_confidence, primary_match_type = self.cross_encoder.map_nli_to_confidence(primary_nli)
            
            # Early stopping: if primary score is too low, skip secondary evaluation
            if primary_confidence < self.score_threshold * 0.5:
                # Still create VerifiedMatch but mark as no_match
                combined_score = primary_confidence
                match_type = "no_match"
                secondary_score = None
            else:
                # Evaluate secondary clauses if they exist
                candidate_clauses = self.cross_encoder.extract_secondary_clauses(
                    candidate.canonical_event.canonical_text
                )
                
                if query_clauses and candidate_clauses:
                    secondary_score = await self.cross_encoder.score_secondary_clauses_async(
                        query_clauses, candidate_clauses
                    )
                else:
                    # No secondary clauses, use primary score only
                    secondary_score = None
                
                # Combine scores
                if secondary_score is not None:
                    combined_score = (
                        self.primary_weight * primary_confidence +
                        self.secondary_weight * secondary_score
                    )
                else:
                    combined_score = primary_confidence
                
                # Determine final match type
                if combined_score >= self.score_threshold:
                    if primary_confidence > self.cross_encoder.entailment_threshold:
                        match_type = "full_match"
                    else:
                        match_type = "partial_match"
                else:
                    match_type = "no_match"
            
            # Create VerifiedMatch
            verified_match = VerifiedMatch(
                candidate_match=candidate,
                cross_encoder_score=combined_score,
                match_type=match_type,
                nli_scores=primary_nli,
                primary_event_score=primary_confidence,
                secondary_clause_score=secondary_score,
                verification_metadata={
                    "primary_match_type": primary_match_type,
                    "model": self.cross_encoder.model_name,
                }
            )
            
            verified_matches.append(verified_match)
        
        # Filter by threshold and sort by confidence
        filtered_matches = [
            vm for vm in verified_matches
            if vm.cross_encoder_score >= self.score_threshold
        ]
        
        # Sort by confidence (highest first)
        filtered_matches.sort(key=lambda x: x.cross_encoder_score, reverse=True)
        
        # Return top-k
        return filtered_matches[:self.top_k]

