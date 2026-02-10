"""
Main ContractSpec extractor with all optimizations.

Orchestrates rule-based extraction with LLM fallback.
Implements parallel field extraction, caching, and early exit.
"""

import asyncio
import hashlib
import logging
from typing import Optional, Dict, Tuple, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.contract_spec import ContractSpec, DateSpec
from .parsers import (
    SectionParser,
    DateParser,
    EntityExtractor,
    ThresholdExtractor,
    DataSourceExtractor
)
from .llm_fallback import LLMFallback
from .circuit_breaker import CircuitBreakerOpenError
from embedding.cache.in_memory import InMemoryCache
import config

logger = logging.getLogger(__name__)


class ContractSpecExtractor:
    """
    Extracts ContractSpec from canonical text with optimizations.
    
    Strategy:
    1. Rule-based extraction (parallel, fast)
    2. Confidence scoring per field
    3. Early exit if confidence >= 0.9
    4. LLM fallback if confidence < threshold
    5. Caching for repeated extractions
    """
    
    def __init__(
        self,
        use_llm_fallback: bool = True,
        confidence_threshold: float = 0.7,
        high_confidence_threshold: float = 0.9,
        track_evidence_spans: bool = False,
        cache: Optional[InMemoryCache] = None
    ):
        """
        Initialize extractor.
        
        Args:
            use_llm_fallback: Enable LLM fallback for low confidence
            confidence_threshold: Minimum confidence for rule-based (0.0-1.0)
            high_confidence_threshold: Early exit threshold (skip LLM)
            track_evidence_spans: Track character spans (slower, for debugging)
            cache: Optional cache for extracted specs
        """
        self.use_llm_fallback = use_llm_fallback
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.track_evidence_spans = track_evidence_spans
        self.cache = cache or InMemoryCache(max_size=config.EXTRACTION_CACHE_MAX_SIZE)
        
        self.section_parser = SectionParser()
        self.date_parser = DateParser()
        self.entity_extractor = EntityExtractor()
        self.threshold_extractor = ThresholdExtractor()
        self.data_source_extractor = DataSourceExtractor()
        self.llm_fallback = LLMFallback() if use_llm_fallback else None
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        await self.entity_extractor.initialize()
        await self.cache.initialize()
        
        self._initialized = True
        logger.info("ContractSpecExtractor initialized")
    
    async def extract_async(
        self,
        canonical_text: str,
        content_hash: Optional[str] = None
    ) -> ContractSpec:
        """
        Extract ContractSpec from canonical text.
        
        Args:
            canonical_text: Structured markdown text
            content_hash: Optional content hash for caching
            
        Returns:
            ContractSpec object with extracted fields
        """
        if not self._initialized:
            await self.initialize()
        
        if content_hash is None:
            content_hash = hashlib.sha256(canonical_text.encode()).hexdigest()
        
        cached = await self.cache.get(content_hash)
        if cached:
            logger.debug(f"Cache hit for content_hash: {content_hash[:8]}")
            return cached
        
        spec, confidence = await self._extract_rule_based(canonical_text)
        
        if confidence >= self.high_confidence_threshold:
            logger.debug(f"High confidence ({confidence:.2f}), skipping LLM fallback")
            await self.cache.set(content_hash, spec)
            return spec
        
        if self.use_llm_fallback and confidence < self.confidence_threshold:
            try:
                failed_fields = self._get_failed_fields(spec, confidence)
                llm_spec = await self.llm_fallback.extract_with_llm(
                    canonical_text,
                    failed_fields
                )
                await self.cache.set(content_hash, llm_spec)
                return llm_spec
            except CircuitBreakerOpenError:
                logger.warning("LLM fallback unavailable, using rule-based result")
                await self.cache.set(content_hash, spec)
                return spec
        
        await self.cache.set(content_hash, spec)
        return spec
    
    async def _extract_rule_based(
        self,
        canonical_text: str
    ) -> Tuple[ContractSpec, float]:
        """
        Extract using rule-based methods (parallel).
        
        Returns:
            Tuple of (ContractSpec, confidence_score)
        """
        statement_task = self.section_parser.parse_statement(canonical_text)
        criteria_task = self.section_parser.parse_resolution_criteria(canonical_text)
        outcomes_task = self.section_parser.parse_outcomes(canonical_text)
        date_str_task = self.section_parser.parse_end_date(canonical_text)
        
        statement, criteria, outcomes, date_str = await asyncio.gather(
            statement_task, criteria_task, outcomes_task, date_str_task
        )
        
        statement_text, statement_span = statement
        criteria_text, criteria_span = criteria
        outcomes_list, outcomes_spans = outcomes
        date_str_val, date_str_span = date_str
        
        dates_task = self.date_parser.parse_dates(canonical_text, date_str_val)
        entities_task = self.entity_extractor.extract_entities(statement_text, criteria_text)
        thresholds_task = self.threshold_extractor.extract_thresholds(statement_text, criteria_text)
        data_source_task = self.data_source_extractor.extract_data_source(criteria_text)
        
        dates, entities, thresholds, data_source = await asyncio.gather(
            dates_task, entities_task, thresholds_task, data_source_task
        )
        
        resolution_date, event_date = dates
        
        evidence_spans = {}
        if self.track_evidence_spans:
            if statement_span:
                evidence_spans["statement"] = [statement_span]
            if criteria_span:
                evidence_spans["resolution_criteria"] = [criteria_span]
            if date_str_span:
                evidence_spans["resolution_date"] = [date_str_span]
            if outcomes_spans:
                evidence_spans["outcome_labels"] = outcomes_spans
        
        spec = ContractSpec(
            statement=statement_text,
            resolution_date=resolution_date,
            event_date=event_date,
            entities=entities,
            thresholds=thresholds,
            resolution_criteria=criteria_text,
            data_source=data_source,
            outcome_labels=outcomes_list if outcomes_list else ["Yes", "No"],
            evidence_spans=evidence_spans if self.track_evidence_spans else {}
        )
        
        confidence = self._calculate_confidence(spec)
        spec.confidence = confidence
        
        return spec, confidence
    
    def _calculate_confidence(self, spec: ContractSpec) -> float:
        """Calculate confidence score for extraction."""
        scores = []
        
        if spec.statement:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        if spec.resolution_criteria:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        if spec.outcome_labels:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        if spec.resolution_date:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        if spec.entities:
            scores.append(0.8 if len(spec.entities) > 0 else 0.5)
        else:
            scores.append(0.0)
        
        if spec.thresholds:
            scores.append(0.9 if len(spec.thresholds) > 0 else 0.5)
        else:
            scores.append(0.0)
        
        if spec.data_source:
            scores.append(0.7)
        else:
            scores.append(0.0)
        
        return sum(scores) / len(scores)
    
    def _get_failed_fields(
        self,
        spec: ContractSpec,
        confidence: float
    ) -> List[str]:
        """Get list of fields that failed extraction."""
        failed = []
        
        if not spec.statement or len(spec.statement) < 10:
            failed.append("statement")
        if not spec.resolution_date:
            failed.append("resolution_date")
        if not spec.entities:
            failed.append("entities")
        if not spec.thresholds:
            failed.append("thresholds")
        if not spec.data_source:
            failed.append("data_source")
        
        return failed
