"""
Data source extractor using pattern matching.

Fast pattern-based extraction for known data sources.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Known data sources and their patterns
KNOWN_SOURCES = {
    'Coinbase': [r'coinbase', r'cb\.usd'],
    'Binance': [r'binance', r'bnb'],
    'Kraken': [r'kraken'],
    'official results': [r'official\s+results', r'official\s+outcome'],
    'election results': [r'election\s+results', r'electoral'],
    'government data': [r'government', r'federal', r'census'],
}

# Context patterns for data source extraction
SOURCE_CONTEXT_PATTERNS = [
    r'according\s+to\s+([A-Z][a-z]+)',
    r'based\s+on\s+([A-Z][a-z]+)',
    r'from\s+([A-Z][a-z]+)',
    r'per\s+([A-Z][a-z]+)',
]


class DataSourceExtractor:
    """
    Extract data source from resolution criteria.
    
    Uses pattern matching for fast extraction.
    """
    
    async def extract_data_source(
        self,
        resolution_criteria: Optional[str]
    ) -> Optional[str]:
        """
        Extract data source from resolution criteria.
        
        Returns:
            Data source string or None
        """
        if not resolution_criteria:
            return None
        
        text_lower = resolution_criteria.lower()
        
        for source, patterns in KNOWN_SOURCES.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return source
        
        for pattern in SOURCE_CONTEXT_PATTERNS:
            match = re.search(pattern, resolution_criteria, re.IGNORECASE)
            if match:
                potential_source = match.group(1)
                if potential_source in KNOWN_SOURCES:
                    return potential_source
        
        return None

