"""
Threshold extractor with compiled regex patterns and negation detection.

Optimized for speed: pre-compiled patterns, fast matching.
Handles currency ($NNN), percentage (NNN%), and bare numbers near
comparison words (e.g. "over 5.5 points", "reach 100,000").
"""

import re
import logging
from typing import List, Optional, Set, Tuple

from canonicalization.contract_spec import ThresholdSpec

logger = logging.getLogger(__name__)


# Pre-compiled regex patterns (module-level for performance)
CURRENCY_PATTERN = re.compile(r'\$[\d,]+(?:\.\d+)?(?:k|K|m|M)?')
PERCENTAGE_PATTERN = re.compile(r'[\d,]+(?:\.\d+)?%')
COMPARISON_PATTERN = re.compile(r'(above|below|at least|at most|exactly|>|<|>=|<=)')

# Bare-number patterns: comparison word followed by a number, or number
# followed by a known unit.  Year-like integers (1900–2099) are excluded
# to avoid false positives on resolution dates.
_COMPARISON_WORDS = (
    r'(?:above|below|over|under|exceed(?:s|ing)?|reach(?:es|ing)?|'
    r'hit(?:s|ting)?|at\s+least|at\s+most|more\s+than|less\s+than|'
    r'greater\s+than|fewer\s+than)'
)
BARE_NUMBER_AFTER_CMP = re.compile(
    _COMPARISON_WORDS + r'\s+([\d,]+(?:\.\d+)?(?:k|K|m|M)?)',
    re.IGNORECASE,
)
_UNIT_WORDS = (
    r'(?:points?|goals?|yards?|runs?|games?|wins?|losses?|seats?|'
    r'degrees?|cents?|bps|basis\s+points?|rebounds?|assists?|'
    r'touchdowns?|strikeouts?|sacks?|interceptions?|aces?|'
    r'kills?|blocks?)'
)
BARE_NUMBER_BEFORE_UNIT = re.compile(
    r'([\d,]+(?:\.\d+)?)\s+' + _UNIT_WORDS,
    re.IGNORECASE,
)

# Year range used to filter out false-positive bare numbers
_YEAR_RE = re.compile(r'^(?:19|20)\d{2}$')

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
        
        Searches for currency ($NNN), percentage (NNN%), and bare numbers
        near comparison words (e.g. "over 5.5 points").
        
        Returns:
            List of ThresholdSpec objects
        """
        combined = f"{statement} {resolution_criteria or ''}"
        thresholds: List[ThresholdSpec] = []
        # Track (value, unit) pairs already emitted to avoid duplicates
        # when the same number is matched by both currency and bare patterns.
        seen: Set[Tuple[float, str]] = set()
        
        # --- Currency ($NNN) ---
        for match in CURRENCY_PATTERN.finditer(combined):
            threshold = self._parse_currency_match(match, combined)
            if threshold:
                key = (threshold.value, threshold.unit)
                if key not in seen:
                    seen.add(key)
                    thresholds.append(threshold)
        
        # --- Percentage (NNN%) ---
        for match in PERCENTAGE_PATTERN.finditer(combined):
            threshold = self._parse_percentage_match(match, combined)
            if threshold:
                key = (threshold.value, threshold.unit)
                if key not in seen:
                    seen.add(key)
                    thresholds.append(threshold)
        
        # --- Bare numbers near comparison words ---
        for match in BARE_NUMBER_AFTER_CMP.finditer(combined):
            threshold = self._parse_bare_number(match, 1, combined)
            if threshold:
                key = (threshold.value, threshold.unit)
                if key not in seen:
                    seen.add(key)
                    thresholds.append(threshold)
        
        # --- Bare numbers before unit words ---
        for match in BARE_NUMBER_BEFORE_UNIT.finditer(combined):
            # Filter out false positives: game/map/match/win identifiers (e.g., "Game 3 Winner", "Map 1", "3 Winner")
            # These are ordinal identifiers, not thresholds
            match_start = match.start()
            match_end = match.end()
            context_before = combined[max(0, match_start - 20):match_start].lower()
            context_after = combined[match_end:min(len(combined), match_end + 20)].lower()
            matched_text = match.group(0).lower()
            
            # Parse threshold first to check the unit
            threshold = self._parse_bare_number(match, 1, combined)
            if not threshold:
                continue
            
            # Skip if it's a game/map/match/win identifier (ordinal, not threshold)
            # Pattern: "Game X Winner", "Map X", "Match X", "3 Winner" (from "Game 3 Winner")
            # Check if unit is game/map/match/win AND context has winner/loser/etc.
            is_ordinal_identifier = (
                threshold.unit in ['game', 'games', 'map', 'maps', 'match', 'matches', 'win', 'wins']
                and any(keyword in context_after for keyword in ['winner', 'loser', 'victor', 'champion', 'result', 'outcome', 'ner', 'ser'])
            )
            
            # Also check if "Game X", "Map X", "Match X" appears before the number
            is_game_map_match_before = (
                any(prefix in context_before for prefix in ['game ', 'map ', 'match '])
                and threshold.unit in ['game', 'games', 'map', 'maps', 'match', 'matches', 'win', 'wins']
                and any(keyword in context_after for keyword in ['winner', 'loser', 'victor', 'champion', 'result', 'outcome'])
            )
            
            if is_ordinal_identifier or is_game_map_match_before:
                continue
            
            key = (threshold.value, threshold.unit)
            if key not in seen:
                seen.add(key)
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
    
    def _parse_bare_number(
        self,
        match: re.Match,
        group_idx: int,
        text: str
    ) -> Optional[ThresholdSpec]:
        """
        Parse a bare number captured by comparison-word or unit-word patterns.
        
        Filters out year-like integers (1900–2099) to avoid false positives.
        
        Args:
            match: Regex match object.
            group_idx: Capture group index containing the numeric string.
            text: Full combined text for context extraction.
            
        Returns:
            ThresholdSpec or None if the number looks like a year.
        """
        raw = match.group(group_idx).replace(',', '')
        
        multiplier = 1
        if raw.lower().endswith('k'):
            multiplier = 1_000
            raw = raw[:-1]
        elif raw.lower().endswith('m'):
            multiplier = 1_000_000
            raw = raw[:-1]
        
        try:
            value = float(raw) * multiplier
        except ValueError:
            return None
        
        # Skip year-like integers (e.g. 2025, 2026)
        int_str = raw.split('.')[0]
        if _YEAR_RE.match(int_str) and multiplier == 1:
            return None
        
        # Infer unit from surrounding context
        unit = self._infer_unit(match.start(), text)
        
        comparison = self._extract_comparison(match.start(), text)
        is_negated, negation_context = self._detect_negation(text, match.start())
        
        return ThresholdSpec(
            value=value,
            unit=unit,
            comparison=comparison,
            is_negated=is_negated,
            negation_context=negation_context,
        )
    
    def _infer_unit(self, position: int, text: str) -> str:
        """
        Infer the unit from context around the number position.
        
        Returns:
            Unit string (e.g. "points", "goals") or "numeric" as fallback.
        """
        context_end = min(len(text), position + 80)
        after = text[position:context_end].lower()
        
        unit_match = re.search(
            r'[\d,.]+[kKmM]?\s+(' + _UNIT_WORDS + r')',
            after,
            re.IGNORECASE,
        )
        if unit_match:
            return unit_match.group(1).rstrip('s')  # normalize plural
        
        return "numeric"
    
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

