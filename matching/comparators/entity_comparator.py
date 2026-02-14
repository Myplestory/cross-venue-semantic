"""
Entity comparator for pair verification.

Compares entities with domain-specific alias matching and Jaccard
token-overlap fallback.  The alias dictionary covers crypto, politics,
sports, and financial entities commonly found on prediction markets.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from canonicalization.contract_spec import EntitySpec
from .entity_aliases import ALIAS_LOOKUP

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def _canonicalize(name: str) -> str:
    """
    Map a surface-form entity name to its canonical name via the alias
    dictionary.  Falls back to the lowered input when there is no alias.
    """
    return ALIAS_LOOKUP.get(name.lower(), name.lower())


def _token_overlap(a: str, b: str) -> float:
    """
    Jaccard similarity over whitespace tokens (case-insensitive).

    Returns 0.0–1.0.  Used as a soft fallback when the alias dictionary
    has no entry for either entity.
    """
    tokens_a: Set[str] = set(a.lower().split())
    tokens_b: Set[str] = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


# ── Comparator ────────────────────────────────────────────────────────

class EntityComparator:
    """
    Compare entities with domain-specific alias matching and fuzzy
    token overlap.

    Matching strategy (per entity pair):
    1. Canonicalize both names via the alias dictionary.
       If canonical forms match -> exact match (score 1.0).
    2. Otherwise, compute Jaccard token overlap.
       If overlap >= FUZZY_THRESHOLD -> fuzzy match (score = overlap).
    3. Otherwise -> no match (score 0.0).

    The overall score is ``matched / max(len_a, len_b)``.
    """

    # Minimum Jaccard token overlap to count as a fuzzy match.
    FUZZY_THRESHOLD: float = 0.5

    def __init__(self):
        """Initialize entity comparator."""
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize comparator (no-op for entity comparator)."""
        self._initialized = True
        logger.debug("EntityComparator initialized")

    async def compare_entities(
        self,
        entities_a: List[EntitySpec],
        entities_b: List[EntitySpec],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compare entity lists using alias + fuzzy matching.

        Returns:
            Tuple of (match_score, comparison_details)
        """
        if not self._initialized:
            await self.initialize()

        # Both empty — neutral; extraction yielded nothing for either side
        if not entities_a and not entities_b:
            return (0.5, {
                "match": False,
                "reason": "both_empty",
                "score": 0.5,
                "matched_count": 0,
                "total_count": 0,
            })

        # One empty
        if not entities_a or not entities_b:
            return (0.5, {
                "match": False,
                "reason": "one_empty",
                "entities_a_count": len(entities_a),
                "entities_b_count": len(entities_b),
                "score": 0.5,
            })

        # ── Build canonical-name -> EntitySpec maps ────────────────
        def _build_map(
            entities: List[EntitySpec],
        ) -> Dict[str, EntitySpec]:
            """Map canonical name -> entity (first occurrence wins)."""
            m: Dict[str, EntitySpec] = {}
            for ent in entities:
                canon = _canonicalize(ent.name)
                if canon not in m:
                    m[canon] = ent
                # Also index any aliases the EntitySpec already carries
                for alias in ent.aliases:
                    c = _canonicalize(alias)
                    if c not in m:
                        m[c] = ent
            return m

        map_a = _build_map(entities_a)
        map_b = _build_map(entities_b)

        # ── Phase 1: Exact alias matching ─────────────────────────
        matched_a_ids: set = set()   # id() of matched A entities
        matched_b_ids: set = set()   # id() of matched B entities
        match_details: list = []

        for canon_a, ent_a in map_a.items():
            if id(ent_a) in matched_a_ids:
                continue
            if canon_a in map_b:
                ent_b = map_b[canon_a]
                if id(ent_b) in matched_b_ids:
                    continue
                matched_a_ids.add(id(ent_a))
                matched_b_ids.add(id(ent_b))
                type_match = ent_a.entity_type == ent_b.entity_type
                match_details.append({
                    "name_a": ent_a.name,
                    "name_b": ent_b.name,
                    "canonical": canon_a,
                    "method": "alias",
                    "type_match": type_match,
                })

        # ── Phase 2: Fuzzy token-overlap for unmatched entities ───
        unmatched_a = [e for e in entities_a if id(e) not in matched_a_ids]
        unmatched_b = [e for e in entities_b if id(e) not in matched_b_ids]

        for ent_a in unmatched_a:
            if id(ent_a) in matched_a_ids:
                continue
            best_score = 0.0
            best_ent_b: Optional[EntitySpec] = None
            for ent_b in unmatched_b:
                if id(ent_b) in matched_b_ids:
                    continue
                overlap = _token_overlap(ent_a.name, ent_b.name)
                if overlap > best_score:
                    best_score = overlap
                    best_ent_b = ent_b
            if best_ent_b is not None and best_score >= self.FUZZY_THRESHOLD:
                matched_a_ids.add(id(ent_a))
                matched_b_ids.add(id(best_ent_b))
                type_match = ent_a.entity_type == best_ent_b.entity_type
                match_details.append({
                    "name_a": ent_a.name,
                    "name_b": best_ent_b.name,
                    "canonical": None,
                    "method": "fuzzy",
                    "fuzzy_score": round(best_score, 3),
                    "type_match": type_match,
                })

        # ── Score ────────────────────────────────────────────────
        total = max(len(entities_a), len(entities_b))
        matched = len(matched_a_ids)
        score = matched / total if total > 0 else 1.0

        # Penalize entity-type mismatches
        type_mismatches = sum(1 for m in match_details if not m["type_match"])
        if type_mismatches > 0:
            score = max(0.0, score - (type_mismatches / total) * 0.3)

        details = {
            "match": score >= 0.8,
            "matched_count": matched,
            "total_count": total,
            "entities_a_count": len(entities_a),
            "entities_b_count": len(entities_b),
            "type_mismatches": type_mismatches,
            "match_details": match_details,
            "score": score,
        }

        return (score, details)
