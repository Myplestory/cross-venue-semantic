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

        Three-phase batch scoring:
        1. All primary events scored bidirectionally in one batch GPU call
           (forward + reverse pairs merged via min-entailment / max-contradiction)
        2. All secondary clause pairs across qualifying candidates
           collected and scored in one batch GPU call

        Total GPU batch calls: 2 (regardless of candidate count).

        Args:
            query_event: Query market canonical event
            candidates: List of candidate matches from retrieval

        Returns:
            List of verified matches, sorted by confidence (highest first)
        """
        if not candidates:
            return []

        query_text = query_event.canonical_text
        query_primary = self.cross_encoder.extract_primary_event(query_text)
        query_clauses = self.cross_encoder.extract_secondary_clauses(query_text)

        candidate_texts = [c.canonical_event.canonical_text for c in candidates]
        candidate_primaries = [
            self.cross_encoder.extract_primary_event(text) for text in candidate_texts
        ]

        # Phase 1: bidirectional batch score all primary events
        # Score both (query→candidate) and (candidate→query) in one GPU
        # call, then merge via min(entailment) / max(contradiction) to
        # ensure mutual entailment — catches subset/superset false positives.
        n = len(candidates)
        forward_pairs = [(query_primary, cp) for cp in candidate_primaries]
        reverse_pairs = [(cp, query_primary) for cp in candidate_primaries]
        all_primary_nli = await self.cross_encoder.score_batch_async(
            forward_pairs + reverse_pairs
        )
        primary_nli_scores = [
            self.cross_encoder.merge_bidirectional(
                all_primary_nli[i], all_primary_nli[n + i]
            )
            for i in range(n)
        ]

        # Phase 2: identify candidates needing secondary scoring
        primary_results = []
        secondary_needed_indices: List[int] = []
        candidate_clauses_map: dict[int, List[str]] = {}

        for i in range(len(candidates)):
            primary_nli = primary_nli_scores[i]
            primary_confidence, primary_match_type = (
                self.cross_encoder.map_nli_to_confidence(primary_nli)
            )
            primary_results.append((primary_confidence, primary_match_type, primary_nli))

            # Only evaluate secondary clauses above early-stopping threshold
            if primary_confidence >= self.score_threshold * 0.5 and query_clauses:
                cand_clauses = self.cross_encoder.extract_secondary_clauses(
                    candidates[i].canonical_event.canonical_text
                )
                if cand_clauses:
                    secondary_needed_indices.append(i)
                    candidate_clauses_map[i] = cand_clauses

        # Phase 3: batch score ALL secondary clause pairs in one GPU call
        secondary_scores: dict[int, float] = {}

        if secondary_needed_indices and query_clauses:
            all_clause_pairs: List[tuple[str, str]] = []
            # (candidate_idx, n_query_clauses, n_candidate_clauses) for result distribution
            pair_layout: List[tuple[int, int, int]] = []

            for idx in secondary_needed_indices:
                cand_clauses = candidate_clauses_map[idx]
                pairs_for_candidate = [
                    (qc, cc) for qc in query_clauses for cc in cand_clauses
                ]
                all_clause_pairs.extend(pairs_for_candidate)
                pair_layout.append((idx, len(query_clauses), len(cand_clauses)))

            # 1 GPU batch call for all secondary clause pairs across all candidates
            all_clause_nli = await self.cross_encoder.score_batch_async(all_clause_pairs)

            # Distribute results: best-match per query clause per candidate
            offset = 0
            for cand_idx, n_query, n_cand in pair_layout:
                n_pairs = n_query * n_cand
                clause_nli_slice = all_clause_nli[offset:offset + n_pairs]
                offset += n_pairs

                clause_scores = []
                for q_idx in range(n_query):
                    best_match = 0.0
                    for c_idx in range(n_cand):
                        flat_idx = q_idx * n_cand + c_idx
                        confidence, _ = self.cross_encoder.map_nli_to_confidence(
                            clause_nli_slice[flat_idx]
                        )
                        best_match = max(best_match, confidence)
                    clause_scores.append(best_match)

                secondary_scores[cand_idx] = (
                    sum(clause_scores) / len(clause_scores) if clause_scores else 0.5
                )

        # Phase 4: assemble VerifiedMatch objects
        verified_matches = []
        for i, candidate in enumerate(candidates):
            primary_confidence, primary_match_type, primary_nli = primary_results[i]

            if primary_confidence < self.score_threshold * 0.5:
                combined_score = primary_confidence
                match_type = "no_match"
                secondary_score = None
            else:
                secondary_score = secondary_scores.get(i)

                if secondary_score is not None:
                    combined_score = (
                        self.primary_weight * primary_confidence
                        + self.secondary_weight * secondary_score
                    )
                else:
                    combined_score = primary_confidence

                if combined_score >= self.score_threshold:
                    if primary_confidence > self.cross_encoder.entailment_threshold:
                        match_type = "full_match"
                    else:
                        match_type = "partial_match"
                else:
                    match_type = "no_match"

            verified_matches.append(VerifiedMatch(
                candidate_match=candidate,
                cross_encoder_score=combined_score,
                match_type=match_type,
                nli_scores=primary_nli,
                primary_event_score=primary_confidence,
                secondary_clause_score=secondary_score,
                verification_metadata={
                    "primary_match_type": primary_match_type,
                    "model": self.cross_encoder.model_name,
                },
            ))

        filtered_matches = [
            vm for vm in verified_matches
            if vm.cross_encoder_score >= self.score_threshold
        ]
        filtered_matches.sort(key=lambda x: x.cross_encoder_score, reverse=True)
        return filtered_matches[:self.top_k]

