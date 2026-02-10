"""
Pair verifier for ContractSpec comparison.

Verifies pair equivalence using research-backed weight distribution.
Includes caching, fast paths, and batch processing optimizations.
"""

import asyncio
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.contract_spec import ContractSpec
from .types import VerifiedMatch, VerifiedPair
from .comparators import (
    EntityComparator,
    ThresholdComparator,
    DateComparator,
    OutcomeMapper
)
import config

logger = logging.getLogger(__name__)


class PairCache:
    """
    Simple in-memory LRU cache for VerifiedPair objects.
    
    Similar to InMemoryCache but for VerifiedPair objects.
    """
    
    def __init__(self, max_size: int = 10000):
        """Initialize pair cache."""
        self.max_size = max_size
        self._cache: Dict[str, VerifiedPair] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._initialized = False
        
        logger.info(f"PairCache initialized: max_size={max_size}")
    
    async def initialize(self) -> None:
        """Initialize cache."""
        self._initialized = True
        logger.debug("PairCache initialized")
    
    async def get(self, cache_key: str) -> Optional[VerifiedPair]:
        """Retrieve cached pair."""
        async with self._lock:
            if cache_key in self._cache:
                # Move to end (most recently used)
                pair = self._cache.pop(cache_key)
                self._cache[cache_key] = pair
                self._hits += 1
                return pair
        
        self._misses += 1
        return None
    
    async def set(self, cache_key: str, pair: VerifiedPair) -> None:
        """Cache pair (LRU eviction)."""
        async with self._lock:
            # Remove if exists
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size and len(self._cache) > 0:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"LRU eviction: removed {oldest_key[:8]}...")
            
            # Add to end (most recently used)
            self._cache[cache_key] = pair
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async def _get_stats():
            async with self._lock:
                total = self._hits + self._misses
                hit_rate = (self._hits / total * 100) if total > 0 else 0.0
                return {
                    "hits": self._hits,
                    "misses": self._misses,
                    "hit_rate": hit_rate,
                    "size": len(self._cache),
                    "max_size": self.max_size
                }
        # Note: This is a sync method, but stats are updated async
        # For now, return current state
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size
        }


class PairVerifier:
    """
    Verifies pair equivalence by comparing ContractSpecs.
    
    Uses research-backed weight distribution:
    - Cross-encoder: 50% (primary semantic signal)
    - Threshold: 20% (critical for financial equivalence)
    - Entity: 15% (structural validation)
    - Date: 10% (temporal validation)
    - Data source: 5% (oracle consistency)
    
    Optimizations:
    - Result caching (LRU, 10K pairs)
    - Fast path for binary markets
    - Aggressive early exit
    - Lazy evaluation (staged checks)
    - Batch processing support
    """
    
    def __init__(
        self,
        entity_tolerance: float = None,
        threshold_tolerance_percent: float = None,
        date_tolerance_days: int = None,
        equivalent_threshold: float = None,
        not_equivalent_threshold: float = None,
        cache: Optional['PairCache'] = None,
        cache_max_size: int = None,
        # Configurable weights (research-backed defaults)
        cross_encoder_weight: float = None,
        threshold_weight: float = None,
        entity_weight: float = None,
        date_weight: float = None,
        data_source_weight: float = None
    ):
        """
        Initialize pair verifier with configurable weights.
        
        Args:
            entity_tolerance: Minimum entity match score (default: from config)
            threshold_tolerance_percent: Percentage tolerance for threshold values (default: from config)
            date_tolerance_days: Days tolerance for date matching (default: from config)
            equivalent_threshold: Minimum score for "equivalent" verdict (default: from config)
            not_equivalent_threshold: Maximum score for "not_equivalent" verdict (default: from config)
            cache: Optional cache instance (creates new if None)
            cache_max_size: Maximum cached pairs (default: from config)
            cross_encoder_weight: Weight for cross-encoder score (default: 0.50)
            threshold_weight: Weight for threshold score (default: 0.20)
            entity_weight: Weight for entity score (default: 0.15)
            date_weight: Weight for date score (default: 0.10)
            data_source_weight: Weight for data source match (default: 0.05)
        """
        # Use config defaults if not provided
        self.entity_tolerance = entity_tolerance or config.VERIFICATION_ENTITY_TOLERANCE
        self.threshold_tolerance_percent = threshold_tolerance_percent or config.VERIFICATION_THRESHOLD_TOLERANCE_PERCENT
        self.date_tolerance_days = date_tolerance_days or config.VERIFICATION_DATE_TOLERANCE_DAYS
        self.equivalent_threshold = equivalent_threshold or config.VERIFICATION_EQUIVALENT_THRESHOLD
        self.not_equivalent_threshold = not_equivalent_threshold or config.VERIFICATION_NOT_EQUIVALENT_THRESHOLD
        
        # Configurable weights (research-backed defaults)
        self.cross_encoder_weight = cross_encoder_weight or config.VERIFICATION_CROSS_ENCODER_WEIGHT
        self.threshold_weight = threshold_weight or config.VERIFICATION_THRESHOLD_WEIGHT
        self.entity_weight = entity_weight or config.VERIFICATION_ENTITY_WEIGHT
        self.date_weight = date_weight or config.VERIFICATION_DATE_WEIGHT
        self.data_source_weight = data_source_weight or config.VERIFICATION_DATA_SOURCE_WEIGHT
        
        # Validate weights sum to 1.0
        total_weight = (
            self.cross_encoder_weight + self.threshold_weight + self.entity_weight +
            self.date_weight + self.data_source_weight
        )
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight}"
            )
        
        # Initialize cache
        cache_max = cache_max_size or config.VERIFICATION_CACHE_MAX_SIZE
        self.cache = cache or PairCache(max_size=cache_max)
        
        # Initialize comparators
        self.entity_comparator = EntityComparator()
        self.threshold_comparator = ThresholdComparator()
        self.date_comparator = DateComparator()
        self.outcome_mapper = OutcomeMapper()
        
        self._initialized = False
        
        logger.info(
            f"PairVerifier initialized: "
            f"cross_encoder_weight={self.cross_encoder_weight}, "
            f"threshold_weight={self.threshold_weight}, "
            f"entity_weight={self.entity_weight}, "
            f"date_weight={self.date_weight}, "
            f"data_source_weight={self.data_source_weight}"
        )
    
    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        await self.entity_comparator.initialize()
        await self.threshold_comparator.initialize()
        await self.date_comparator.initialize()
        await self.outcome_mapper.initialize()
        await self.cache.initialize()
        
        self._initialized = True
        logger.info("PairVerifier initialized")
    
    async def verify_pair_async(
        self,
        verified_match: VerifiedMatch,
        contract_spec_a: ContractSpec,
        contract_spec_b: ContractSpec,
        market_a_id: str,
        market_b_id: str
    ) -> VerifiedPair:
        """
        Verify pair equivalence with caching and optimizations.
        
        Strategy:
        1. Check cache (fast path: <1ms)
        2. Early exit checks (cross-encoder score, critical mismatches)
        3. Fast path for binary markets (direct outcome mapping)
        4. Parallel field comparison (entities, thresholds, dates)
        5. Outcome mapping (binary fast path or semantic)
        6. Verdict determination
        7. Cache result
        
        Args:
            verified_match: VerifiedMatch from reranking phase
            contract_spec_a: ContractSpec for market A
            contract_spec_b: ContractSpec for market B
            market_a_id: Market A UUID
            market_b_id: Market B UUID
            
        Returns:
            VerifiedPair with outcome mapping and verdict
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate cache key (sorted to ensure consistency)
        cache_key = self._generate_cache_key(contract_spec_a, contract_spec_b)
        
        # Check cache first (fast path)
        cached = await self.cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for pair: {cache_key[:16]}")
            return cached
        
        # Early exit: Cross-encoder score too low
        if verified_match.cross_encoder_score < self.not_equivalent_threshold:
            verdict = "not_equivalent"
            confidence = verified_match.cross_encoder_score
            outcome_mapping = {}
            comparison_details = {
                "early_exit": "cross_encoder_score_too_low",
                "score": verified_match.cross_encoder_score
            }
            
            verified_pair = VerifiedPair(
                pair_key=self._generate_pair_key(market_a_id, market_b_id),
                market_a_id=market_a_id,
                market_b_id=market_b_id,
                contract_spec_a=contract_spec_a,
                contract_spec_b=contract_spec_b,
                outcome_mapping=outcome_mapping,
                verdict=verdict,
                confidence=confidence,
                comparison_details=comparison_details
            )
            
            await self.cache.set(cache_key, verified_pair)
            return verified_pair
        
        # Fast path: Binary markets (direct outcome mapping)
        if self._is_binary_market(contract_spec_a, contract_spec_b):
            outcome_mapping = await self.outcome_mapper.map_binary_fast_path(
                contract_spec_a.outcome_labels,
                contract_spec_b.outcome_labels
            )
            
            # Quick comparison for binary markets (parallel)
            entity_task = self.entity_comparator.compare_entities(
                contract_spec_a.entities,
                contract_spec_b.entities
            )
            threshold_task = self.threshold_comparator.compare_thresholds(
                contract_spec_a.thresholds,
                contract_spec_b.thresholds,
                self.threshold_tolerance_percent
            )
            date_task = self.date_comparator.compare_dates(
                contract_spec_a.resolution_date,
                contract_spec_b.resolution_date,
                self.date_tolerance_days
            )
            
            (entity_score, entity_details), (threshold_score, threshold_details), (date_score, date_details) = await asyncio.gather(
                entity_task, threshold_task, date_task
            )
            
            # Data source matching
            data_source_match = (
                contract_spec_a.data_source is not None and
                contract_spec_b.data_source is not None and
                contract_spec_a.data_source.lower() == contract_spec_b.data_source.lower()
            )
            
            # Calculate weighted score
            weighted_score = self._calculate_weighted_score(
                entity_score,
                threshold_score,
                date_score,
                data_source_match,
                verified_match.cross_encoder_score
            )
            
            # Determine verdict
            verdict, confidence = self._determine_verdict(
                entity_score,
                threshold_score,
                date_score,
                weighted_score,
                verified_match.cross_encoder_score
            )
            
            comparison_details = {
                "entity_score": entity_score,
                "threshold_score": threshold_score,
                "date_score": date_score,
                "data_source_match": data_source_match,
                "weighted_score": weighted_score,
                "fast_path": "binary_market"
            }
            
            verified_pair = VerifiedPair(
                pair_key=self._generate_pair_key(market_a_id, market_b_id),
                market_a_id=market_a_id,
                market_b_id=market_b_id,
                contract_spec_a=contract_spec_a,
                contract_spec_b=contract_spec_b,
                outcome_mapping=outcome_mapping,
                verdict=verdict,
                confidence=confidence,
                comparison_details=comparison_details
            )
            
            await self.cache.set(cache_key, verified_pair)
            return verified_pair
        
        # Full comparison (parallel)
        entity_task = self.entity_comparator.compare_entities(
            contract_spec_a.entities,
            contract_spec_b.entities
        )
        threshold_task = self.threshold_comparator.compare_thresholds(
            contract_spec_a.thresholds,
            contract_spec_b.thresholds,
            self.threshold_tolerance_percent
        )
        date_task = self.date_comparator.compare_dates(
            contract_spec_a.resolution_date,
            contract_spec_b.resolution_date,
            self.date_tolerance_days
        )
        
        # Parallel execution
        (entity_score, entity_details), (threshold_score, threshold_details), (date_score, date_details) = await asyncio.gather(
            entity_task, threshold_task, date_task
        )
        
        # Early exit: Critical mismatch
        if entity_score < self.entity_tolerance * 0.5:
            verdict = "not_equivalent"
            confidence = min(entity_score, threshold_score, date_score)
            outcome_mapping = {}
            comparison_details = {
                "early_exit": "critical_entity_mismatch",
                "entity_score": entity_score
            }
            
            verified_pair = VerifiedPair(
                pair_key=self._generate_pair_key(market_a_id, market_b_id),
                market_a_id=market_a_id,
                market_b_id=market_b_id,
                contract_spec_a=contract_spec_a,
                contract_spec_b=contract_spec_b,
                outcome_mapping=outcome_mapping,
                verdict=verdict,
                confidence=confidence,
                comparison_details=comparison_details
            )
            
            await self.cache.set(cache_key, verified_pair)
            return verified_pair
        
        # Outcome mapping (semantic for multi-outcome)
        outcome_mapping = await self.outcome_mapper.map_outcomes(
            contract_spec_a.outcome_labels,
            contract_spec_b.outcome_labels,
            contract_spec_a,
            contract_spec_b
        )
        
        # Data source matching
        data_source_match = (
            contract_spec_a.data_source is not None and
            contract_spec_b.data_source is not None and
            contract_spec_a.data_source.lower() == contract_spec_b.data_source.lower()
        )
        
        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(
            entity_score,
            threshold_score,
            date_score,
            data_source_match,
            verified_match.cross_encoder_score
        )
        
        # Determine verdict
        verdict, confidence = self._determine_verdict(
            entity_score,
            threshold_score,
            date_score,
            weighted_score,
            verified_match.cross_encoder_score
        )
        
        comparison_details = {
            "entity_score": entity_score,
            "threshold_score": threshold_score,
            "date_score": date_score,
            "data_source_match": data_source_match,
            "weighted_score": weighted_score,
            "entity_details": entity_details,
            "threshold_details": threshold_details,
            "date_details": date_details
        }
        
        verified_pair = VerifiedPair(
            pair_key=self._generate_pair_key(market_a_id, market_b_id),
            market_a_id=market_a_id,
            market_b_id=market_b_id,
            contract_spec_a=contract_spec_a,
            contract_spec_b=contract_spec_b,
            outcome_mapping=outcome_mapping,
            verdict=verdict,
            confidence=confidence,
            comparison_details=comparison_details
        )
        
        await self.cache.set(cache_key, verified_pair)
        return verified_pair
    
    async def verify_batch_async(
        self,
        verified_matches: List[VerifiedMatch],
        contract_specs: Dict[str, ContractSpec],
        market_ids: List[Tuple[str, str]]
    ) -> List[VerifiedPair]:
        """
        Verify multiple pairs in parallel (batch processing).
        
        Uses asyncio.gather() for concurrent verification.
        Handles partial failures gracefully.
        
        Args:
            verified_matches: List of VerifiedMatch objects
            contract_specs: Dict mapping market IDs to ContractSpecs
            market_ids: List of (market_a_id, market_b_id) tuples
            
        Returns:
            List of VerifiedPair objects
        """
        if not verified_matches:
            return []
        
        tasks = []
        for i, match in enumerate(verified_matches):
            if i >= len(market_ids):
                logger.warning(f"Missing market IDs for match {i}, skipping")
                continue
            
            market_a_id, market_b_id = market_ids[i]
            spec_a_key = f"{market_a_id}_spec"
            spec_b_key = f"{market_b_id}_spec"
            
            if spec_a_key not in contract_specs or spec_b_key not in contract_specs:
                logger.warning(f"Missing ContractSpecs for pair {i}, skipping")
                continue
            
            task = self.verify_pair_async(
                match,
                contract_specs[spec_a_key],
                contract_specs[spec_b_key],
                market_a_id,
                market_b_id
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        verified_pairs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error verifying pair {i}: {result}")
                continue
            verified_pairs.append(result)
        
        return verified_pairs
    
    def _generate_cache_key(
        self,
        spec_a: ContractSpec,
        spec_b: ContractSpec
    ) -> str:
        """Generate cache key from ContractSpec statements."""
        # Use statement hash as proxy for ContractSpec identity
        key_a = hashlib.sha256(spec_a.statement.encode()).hexdigest()[:16]
        key_b = hashlib.sha256(spec_b.statement.encode()).hexdigest()[:16]
        # Sort to ensure consistency
        return f"{min(key_a, key_b)}_{max(key_a, key_b)}"
    
    def _generate_pair_key(self, market_a_id: str, market_b_id: str) -> str:
        """Generate stable pair key (sorted market IDs)."""
        return f"{min(market_a_id, market_b_id)}_{max(market_a_id, market_b_id)}"
    
    def _is_binary_market(
        self,
        spec_a: ContractSpec,
        spec_b: ContractSpec
    ) -> bool:
        """Check if both markets are binary (Yes/No)."""
        return (
            len(spec_a.outcome_labels) == 2 and
            len(spec_b.outcome_labels) == 2
        )
    
    def _calculate_weighted_score(
        self,
        entity_score: float,
        threshold_score: float,
        date_score: float,
        data_source_match: bool,
        cross_encoder_score: float
    ) -> float:
        """
        Calculate weighted score using research-backed distribution.
        
        Research shows cross-encoder (NLI model) should be primary signal (50%),
        with structural checks serving as validation signals.
        """
        return (
            self.entity_weight * entity_score +
            self.threshold_weight * threshold_score +
            self.date_weight * date_score +
            self.data_source_weight * (1.0 if data_source_match else 0.5) +
            self.cross_encoder_weight * cross_encoder_score  # PRIMARY SIGNAL
        )
    
    def _determine_verdict(
        self,
        entity_score: float,
        threshold_score: float,
        date_score: float,
        weighted_score: float,
        cross_encoder_score: float
    ) -> Tuple[str, float]:
        """
        Determine verdict and confidence using research-backed logic.
        
        Verdicts:
        - equivalent: All critical fields match, high confidence (weighted_score >= 0.9)
        - not_equivalent: Critical mismatch or low confidence (weighted_score < 0.5)
        - needs_review: Ambiguous or partial matches (0.5 <= weighted_score < 0.9)
        """
        # Floating-point tolerance for threshold comparisons
        EPSILON = 1e-9
        
        # Critical mismatch check (early exit)
        if (
            entity_score < self.entity_tolerance * 0.5 - EPSILON or
            threshold_score < 0.3 - EPSILON or
            date_score < 0.3 - EPSILON
        ):
            return ("not_equivalent", weighted_score)
        
        # High confidence equivalent
        if (
            weighted_score >= self.equivalent_threshold - EPSILON and
            entity_score >= self.entity_tolerance - EPSILON and
            threshold_score >= 0.8 - EPSILON and
            date_score >= 0.8 - EPSILON and
            cross_encoder_score >= 0.7 - EPSILON  # Cross-encoder must also be high
        ):
            return ("equivalent", weighted_score)
        
        # Low confidence not equivalent
        if weighted_score < self.not_equivalent_threshold - EPSILON:
            return ("not_equivalent", weighted_score)
        
        # Needs review (ambiguous)
        return ("needs_review", weighted_score)

