"""
Threshold comparator for pair verification.

Compares thresholds with tolerance.
"""

import logging
from typing import List, Tuple, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from canonicalization.contract_spec import ThresholdSpec

logger = logging.getLogger(__name__)


class ThresholdComparator:
    """
    Compare thresholds with tolerance.
    
    Handles:
    - Value matching with percentage tolerance
    - Comparison operator matching
    - Unit conversion (if needed)
    - Negation handling
    """
    
    def __init__(self):
        """Initialize threshold comparator."""
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize comparator (no-op for threshold comparator)."""
        self._initialized = True
        logger.debug("ThresholdComparator initialized")
    
    async def compare_thresholds(
        self,
        thresholds_a: List[ThresholdSpec],
        thresholds_b: List[ThresholdSpec],
        tolerance_percent: float = 0.01
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compare threshold lists.
        
        Args:
            tolerance_percent: Percentage tolerance for value matching (default: 1%)
            
        Returns:
            Tuple of (match_score, comparison_details)
        """
        if not self._initialized:
            await self.initialize()
        
        # Both empty
        if not thresholds_a and not thresholds_b:
            return (1.0, {
                "match": True,
                "reason": "both_empty",
                "score": 1.0,
                "matched_count": 0,
                "total_count": 0
            })
        
        # One empty
        if not thresholds_a or not thresholds_b:
            return (0.0, {
                "match": False,
                "reason": "one_empty",
                "thresholds_a_count": len(thresholds_a),
                "thresholds_b_count": len(thresholds_b),
                "score": 0.0
            })
        
        # Match thresholds by unit and comparison
        matched = 0
        total = max(len(thresholds_a), len(thresholds_b))
        match_details = []
        
        for threshold_a in thresholds_a:
            best_match = None
            best_score = 0.0
            
            for threshold_b in thresholds_b:
                # Check unit match
                unit_match = (
                    threshold_a.unit == threshold_b.unit or
                    (threshold_a.unit is None and threshold_b.unit is None)
                )
                
                if not unit_match:
                    continue
                
                # Check comparison operator match
                comparison_match = threshold_a.comparison == threshold_b.comparison
                
                if not comparison_match:
                    continue
                
                # Check negation consistency
                negation_match = threshold_a.is_negated == threshold_b.is_negated
                
                if not negation_match:
                    continue
                
                # Calculate value match score
                value_diff = abs(threshold_a.value - threshold_b.value)
                value_tolerance = abs(threshold_a.value) * tolerance_percent
                
                if value_diff <= value_tolerance:
                    value_score = 1.0
                elif value_diff <= abs(threshold_a.value) * 0.05:
                    value_score = 0.8
                elif value_diff <= abs(threshold_a.value) * 0.10:
                    value_score = 0.5
                else:
                    value_score = 0.0
                
                # Combined score
                score = value_score
                if score > best_score:
                    best_score = score
                    best_match = threshold_b
            
            if best_match and best_score > 0.5:
                matched += 1
                match_details.append({
                    "value_a": threshold_a.value,
                    "value_b": best_match.value,
                    "unit": threshold_a.unit,
                    "comparison": threshold_a.comparison,
                    "value_diff": abs(threshold_a.value - best_match.value),
                    "score": best_score
                })
        
        # Calculate match score
        if total == 0:
            score = 1.0
        else:
            score = matched / total
        
        details = {
            "match": score >= 0.8,
            "matched_count": matched,
            "total_count": total,
            "thresholds_a_count": len(thresholds_a),
            "thresholds_b_count": len(thresholds_b),
            "match_details": match_details,
            "score": score
        }
        
        return (score, details)

