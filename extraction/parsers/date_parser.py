"""
Date parser with fast path for ISO dates.

Optimized for common date formats with fallback to dateutil.
"""

import re
import logging
from typing import Optional, Tuple
from datetime import datetime

try:
    from dateutil import parser as dateutil_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    logging.warning("python-dateutil not available, date parsing may be limited")

from canonicalization.contract_spec import DateSpec

logger = logging.getLogger(__name__)


# Fast path regex for ISO format (YYYY-MM-DD)
ISO_DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')


class DateParser:
    """
    Extract dates with context analysis.
    
    Fast path for ISO dates, fallback to dateutil for complex formats.
    """
    
    async def parse_dates(
        self,
        canonical_text: str,
        end_date_str: Optional[str]
    ) -> Tuple[Optional[DateSpec], Optional[DateSpec]]:
        """
        Extract resolution_date and event_date.
        
        Logic:
        - If "End Date:" exists → resolution_date with is_deadline=True
        - If statement mentions event date → event_date
        - Context analysis: "by", "before" → deadline
                          "on", "at", "when" → event date
        
        Returns:
            Tuple of (resolution_date, event_date)
        """
        resolution_date = None
        event_date = None
        
        if end_date_str:
            try:
                date = self._parse_date_fast(end_date_str)
                if date:
                    resolution_date = DateSpec(
                        date=date,
                        is_deadline=True
                    )
            except Exception as e:
                logger.warning(f"Failed to parse end date '{end_date_str}': {e}")
        
        event_date = self._extract_event_date(canonical_text)
        
        return resolution_date, event_date
    
    def _parse_date_fast(self, date_str: str) -> Optional[datetime]:
        """
        Fast date parsing with ISO format optimization.
        
        Returns:
            Parsed datetime or None
        """
        date_str = date_str.strip()
        
        if ISO_DATE_PATTERN.match(date_str):
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                pass
        
        if HAS_DATEUTIL:
            try:
                return dateutil_parser.parse(date_str)
            except Exception:
                pass
        
        return None
    
    def _extract_event_date(self, canonical_text: str) -> Optional[DateSpec]:
        """
        Extract event date from statement context.
        
        Looks for patterns like "on Dec 31, 2024" (event date, not deadline).
        """
        lines = canonical_text.split('\n')
        statement = lines[0] if lines else ""
        
        event_patterns = [
            r'on\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'when\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'at\s+(\w+\s+\d{1,2},?\s+\d{4})',
        ]
        
        for pattern in event_patterns:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                date = self._parse_date_fast(date_str)
                if date:
                    return DateSpec(
                        date=date,
                        is_deadline=False
                    )
        
        return None

