"""
Outcome mapper for pair verification.

Maps outcomes between venues with fast path for binary markets.
"""

import logging
from typing import List, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from canonicalization.contract_spec import ContractSpec

logger = logging.getLogger(__name__)


class OutcomeMapper:
    """
    Map outcomes between venues with fast path for binary markets.
    
    Handles:
    - Binary markets (Yes/No) - fast path
    - Multi-outcome markets - semantic matching
    - Venue-specific label variations
    """
    
    def __init__(self):
        """Initialize outcome mapper."""
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize mapper (no-op for outcome mapper)."""
        self._initialized = True
        logger.debug("OutcomeMapper initialized")
    
    async def map_binary_fast_path(
        self,
        outcomes_a: List[str],
        outcomes_b: List[str]
    ) -> Dict[str, str]:
        """
        Fast path for binary markets (direct mapping).
        
        Maps: Yes ↔ YES, No ↔ NO (case-insensitive)
        
        Args:
            outcomes_a: Outcome labels from market A
            outcomes_b: Outcome labels from market B
            
        Returns:
            Outcome mapping dict: {"YES_A": "YES_B", "NO_A": "NO_B"}
        """
        if not self._initialized:
            await self.initialize()
        
        # Normalize labels
        norm_a = [o.lower().strip() for o in outcomes_a]
        norm_b = [o.lower().strip() for o in outcomes_b]
        
        mapping = {}
        
        # Find Yes/YES match
        yes_a = None
        yes_b = None
        for i, o in enumerate(norm_a):
            if o in ["yes", "y", "true", "1"]:
                yes_a = outcomes_a[i]
                break
        for i, o in enumerate(norm_b):
            if o in ["yes", "y", "true", "1"]:
                yes_b = outcomes_b[i]
                break
        
        if yes_a and yes_b:
            mapping[yes_a] = yes_b
        
        # Find No/NO match
        no_a = None
        no_b = None
        for i, o in enumerate(norm_a):
            if o in ["no", "n", "false", "0"]:
                no_a = outcomes_a[i]
                break
        for i, o in enumerate(norm_b):
            if o in ["no", "n", "false", "0"]:
                no_b = outcomes_b[i]
                break
        
        if no_a and no_b:
            mapping[no_a] = no_b
        
        return mapping
    
    async def map_outcomes(
        self,
        outcomes_a: List[str],
        outcomes_b: List[str],
        contract_spec_a: Optional[ContractSpec] = None,
        contract_spec_b: Optional[ContractSpec] = None
    ) -> Dict[str, str]:
        """
        Map outcomes between two markets.
        
        Uses fast path for binary markets, semantic matching for multi-outcome.
        
        Args:
            outcomes_a: Outcome labels from market A
            outcomes_b: Outcome labels from market B
            contract_spec_a: Optional ContractSpec A (for future semantic matching)
            contract_spec_b: Optional ContractSpec B (for future semantic matching)
            
        Returns:
            Outcome mapping dict: {"OUTCOME_A": "OUTCOME_B", ...}
        """
        if not self._initialized:
            await self.initialize()
        
        # Fast path for binary markets
        if len(outcomes_a) == 2 and len(outcomes_b) == 2:
            return await self.map_binary_fast_path(outcomes_a, outcomes_b)
        
        # Semantic matching for multi-outcome (future: use embedding similarity)
        # For now, use order-based mapping with label normalization
        mapping = {}
        for i, outcome_a in enumerate(outcomes_a):
            if i < len(outcomes_b):
                mapping[outcome_a] = outcomes_b[i]
        
        return mapping

