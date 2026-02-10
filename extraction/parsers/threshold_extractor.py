"""
Threshold extractor with compiled regex patterns and negation detection.

Optimized for speed: pre-compiled patterns, fast matching.
"""

import re
import logging
from typing import List, Optional, Tuple

from canonicalization.contract_spec import ThresholdSpec

logger = logging.getLogger(__name__)


# Pre-compiled regex patterns (module-level for performance)
CURRENCY_PATTERN = re.compile(r'\$[\d,]+(?:\.\d+)?(?:k|K|m|M)?')
PERCENTAGE_PATTERN = re.compile(r'[\d,]+(?:\.\d+)?%')
COMPARISON_PATTERN = re.compile(r'(above|below|at least|at most|exactly|>|<|>=|<=)')
NEGATION_PATTERNS = [
    re.compile(r'not\s+(?:to\s+)?exceed', re.IGNORECASE),
    re.compile(r'not\s+(?:to\s+)?(?:go\s+)?above', re.IGNORECASE),
    re.compile(r'not\s+(?:to\s+)?(?:go\s+)?below', re.IGNORECASE),
    re.compile(r'unless\s+(?:it\s+)?(?:is\s+)?(?:above|below|exceeds?)', re.IGNORECASE),
    re.compile(r'except\s+(?:if|when|where)', re.IGNORECASE),
    re.compile(r'no\s+more\s+than', re.IGNORECASE),
    re.compile(r'less\s+than', re.IGNORECASE),
]


class ThresholdExtractor:
    """
    Extract numeric thresholds with comparison operators and negation detection.
    
    Uses pre-compiled regex patterns for fast matching.
    """
    
    async def extract_thresholds(
        self,
        statement: str,
        resolution_criteria: Optional[str]
    ) -> List[ThresholdSpec]:
        """
        Extract thresholds from text.
        
        Returns:
            List of ThresholdSpec objects
        """
        combined = f"{statement} {resolution_criteria or ''}"
        thresholds = []
        
        currency_matches = CURRENCY_PATTERN.finditer(combined)
        percentage_matches = PERCENTAGE_PATTERN.finditer(combined)
        
        for match in currency_matches:
            threshold = self._parse_currency_match(match, combined)
            if threshold:
                thresholds.append(threshold)
        
        for match in percentage_matches:
            threshold = self._parse_percentage_match(match, combined)
            if threshold:
                thresholds.append(threshold)
        
        return thresholds
    
    def _parse_currency_match(
        self,
        match: re.Match,
        text: str
    ) -> Optional[ThresholdSpec]:
        """Parse currency threshold match."""
        value_str = match.group(0).replace('$', '').replace(',', '')
        
        multiplier = 1
        if value_str.lower().endswith('k'):
            multiplier = 1000
            value_str = value_str[:-1]
        elif value_str.lower().endswith('m'):
            multiplier = 1000000
            value_str = value_str[:-1]
        
        try:
            value = float(value_str) * multiplier
        except ValueError:
            return None
        
        comparison = self._extract_comparison(match.start(), text)
        is_negated, negation_context = self._detect_negation(
            text, match.start()
        )
        
        return ThresholdSpec(
            value=value,
            unit="dollars",
            comparison=comparison,
            is_negated=is_negated,
            negation_context=negation_context
        )
    
    def _parse_percentage_match(
        self,
        match: re.Match,
        text: str
    ) -> Optional[ThresholdSpec]:
        """Parse percentage threshold match."""
        value_str = match.group(0).replace('%', '').replace(',', '')
        
        try:
            value = float(value_str)
        except ValueError:
            return None
        
        comparison = self._extract_comparison(match.start(), text)
        is_negated, negation_context = self._detect_negation(
            text, match.start()
        )
        
        return ThresholdSpec(
            value=value,
            unit="percentage",
            comparison=comparison,
            is_negated=is_negated,
            negation_context=negation_context
        )
    
    def _extract_comparison(self, position: int, text: str) -> str:
        """Extract comparison operator near threshold position."""
        context_start = max(0, position - 30)
        context = text[context_start:position + 50].lower()
        
        comparison_match = COMPARISON_PATTERN.search(context)
        if comparison_match:
            op = comparison_match.group(1)
            mapping = {
                'above': '>',
                'below': '<',
                'at least': '>=',
                'at most': '<=',
                'exactly': '==',
            }
            return mapping.get(op, op)
        
        return ">="
    
    def _detect_negation(
        self,
        text: str,
        threshold_position: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if threshold has negation modifier.
        
        Returns:
            Tuple of (is_negated, negation_context)
        """
        context_start = max(0, threshold_position - 50)
        context_text = text[context_start:threshold_position].lower()
        
        for pattern in NEGATION_PATTERNS:
            if pattern.search(context_text):
                negation_context = self._extract_negation_context(
                    text, threshold_position, pattern
                )
                return True, negation_context
        
        return False, None
    
    def _extract_negation_context(
        self,
        text: str,
        threshold_position: int,
        pattern: re.Pattern
    ) -> Optional[str]:
        """Extract context around negation."""
        context_start = max(0, threshold_position - 100)
        context_end = min(len(text), threshold_position + 50)
        return text[context_start:context_end].strip()

