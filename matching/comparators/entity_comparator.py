"""
Entity comparator for pair verification.

Compares entities with alias matching.
"""

import logging
from typing import List, Tuple, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from canonicalization.contract_spec import EntitySpec

logger = logging.getLogger(__name__)


class EntityComparator:
    """
    Compare entities with alias matching.
    
    Handles:
    - Exact name matching
    - Alias matching (e.g., "Bitcoin" → "BTC")
    - Entity type matching
    - Partial entity sets (some entities may be missing)
    """
    
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
        entities_b: List[EntitySpec]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compare entity lists.
        
        Returns:
            Tuple of (match_score, comparison_details)
        """
        if not self._initialized:
            await self.initialize()
        
        # Both empty
        if not entities_a and not entities_b:
            return (1.0, {
                "match": True,
                "reason": "both_empty",
                "score": 1.0,
                "matched_count": 0,
                "total_count": 0
            })
        
        # One empty
        if not entities_a or not entities_b:
            return (0.5, {
                "match": False,
                "reason": "one_empty",
                "entities_a_count": len(entities_a),
                "entities_b_count": len(entities_b),
                "score": 0.5
            })
        
        # Build entity name sets (with aliases)
        def build_entity_set(entities: List[EntitySpec]) -> Dict[str, EntitySpec]:
            """Build entity set with all names and aliases."""
            entity_set = {}
            for entity in entities:
                # Add primary name
                entity_set[entity.name.lower()] = entity
                # Add aliases
                for alias in entity.aliases:
                    entity_set[alias.lower()] = entity
            return entity_set
        
        set_a = build_entity_set(entities_a)
        set_b = build_entity_set(entities_b)
        
        # Find matches — track by object identity to avoid double-counting
        # when aliases create multiple keys for the same entity
        matched_a_ids = set()  # id() of matched entities from A
        matched_b_ids = set()  # id() of matched entities from B
        total = max(len(entities_a), len(entities_b))
        match_details = []
        
        for name_a, entity_a in set_a.items():
            if name_a in set_b:
                entity_b = set_b[name_a]
                # Skip if either entity already matched (alias overlap)
                if id(entity_a) in matched_a_ids or id(entity_b) in matched_b_ids:
                    continue
                matched_a_ids.add(id(entity_a))
                matched_b_ids.add(id(entity_b))
                # Check entity type match
                type_match = entity_a.entity_type == entity_b.entity_type
                match_details.append({
                    "name": entity_a.name,
                    "type_match": type_match,
                    "entity_type_a": entity_a.entity_type,
                    "entity_type_b": entity_b.entity_type
                })
        
        # Calculate match score
        matched = len(matched_a_ids)
        if total == 0:
            score = 1.0
        else:
            score = matched / total
        
        # Penalize entity type mismatches
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
            "score": score
        }
        
        return (score, details)

