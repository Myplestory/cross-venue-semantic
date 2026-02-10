"""
Date comparator for pair verification.

Compares date specifications with tolerance.
"""

import logging
from typing import Tuple, Dict, Any, Optional
from datetime import timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from canonicalization.contract_spec import DateSpec

logger = logging.getLogger(__name__)


class DateComparator:
    """
    Compare dates with tolerance.
    
    Handles:
    - Resolution date matching (with day tolerance)
    - Event date matching
    - Deadline vs event date distinction
    """
    
    def __init__(self):
        """Initialize date comparator."""
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize comparator (no-op for date comparator)."""
        self._initialized = True
        logger.debug("DateComparator initialized")
    
    async def compare_dates(
        self,
        date_a: Optional[DateSpec],
        date_b: Optional[DateSpec],
        tolerance_days: int = 1
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compare date specifications.
        
        Args:
            date_a: First date specification
            date_b: Second date specification
            tolerance_days: Days tolerance for matching (default: 1 day)
            
        Returns:
            Tuple of (match_score, comparison_details)
        """
        if not self._initialized:
            await self.initialize()
        
        # Both dates missing
        if date_a is None and date_b is None:
            return (0.5, {
                "match": True,
                "reason": "both_missing",
                "score": 0.5
            })
        
        # One date missing
        if date_a is None or date_b is None:
            return (0.0, {
                "match": False,
                "reason": "one_missing",
                "date_a": date_a.date.isoformat() if date_a else None,
                "date_b": date_b.date.isoformat() if date_b else None,
                "score": 0.0
            })
        
        # Calculate date difference
        date_diff = abs((date_a.date - date_b.date).days)
        
        # Check is_deadline consistency
        deadline_match = date_a.is_deadline == date_b.is_deadline
        
        # Calculate match score
        if date_diff == 0:
            score = 1.0
        elif date_diff <= tolerance_days:
            score = 0.9
        elif date_diff <= 7:
            score = 0.7
        else:
            score = 0.0
        
        # Penalize deadline mismatch
        if not deadline_match:
            score = min(score, 0.5)
        
        details = {
            "match": score >= 0.7,
            "date_a": date_a.date.isoformat(),
            "date_b": date_b.date.isoformat(),
            "date_diff_days": date_diff,
            "deadline_match": deadline_match,
            "is_deadline_a": date_a.is_deadline,
            "is_deadline_b": date_b.is_deadline,
            "score": score
        }
        
        return (score, details)

