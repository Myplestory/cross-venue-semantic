"""
Data source extractor using pattern matching.

Fast pattern-based extraction for known data sources.
Searches both statement and resolution criteria for
exchange names, league names, and generic provenance phrases.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Known data sources and their patterns.
# Keys are the canonical source name returned to callers.
KNOWN_SOURCES = {
    # Crypto / financial exchanges
    'Coinbase': [r'coinbase', r'cb\.usd'],
    'Binance': [r'binance'],
    'Kraken': [r'kraken'],
    'CoinGecko': [r'coingecko'],
    'CoinMarketCap': [r'coinmarketcap', r'cmc'],
    'Bloomberg': [r'bloomberg'],
    'Reuters': [r'reuters'],
    'Yahoo Finance': [r'yahoo\s+finance'],
    'TradingView': [r'tradingview'],
    # Sports leagues / official bodies
    'NBA': [r'\bNBA\b'],
    'NFL': [r'\bNFL\b'],
    'MLB': [r'\bMLB\b'],
    'NHL': [r'\bNHL\b'],
    'MLS': [r'\bMLS\b'],
    'FIFA': [r'\bFIFA\b'],
    'UEFA': [r'\bUEFA\b'],
    'ATP': [r'\bATP\b'],
    'WTA': [r'\bWTA\b'],
    'PGA': [r'\bPGA\b'],
    'ICC': [r'\bICC\b'],
    # News / data providers
    'ESPN': [r'\bESPN\b'],
    'AP': [r'\bAssociated\s+Press\b', r'\bAP\s+News\b'],
    'FiveThirtyEight': [r'fivethirtyeight', r'538'],
    'RealClearPolitics': [r'realclearpolitics', r'rcp'],
    # Government / institutional
    'official results': [r'official\s+results', r'official\s+outcome'],
    'election results': [r'election\s+results', r'electoral'],
    'government data': [r'government\s+data', r'\bfederal\b', r'\bcensus\b'],
    'BLS': [r'\bBLS\b', r'Bureau\s+of\s+Labor'],
    'BEA': [r'\bBEA\b', r'Bureau\s+of\s+Economic'],
    'NOAA': [r'\bNOAA\b'],
}

# Context patterns — capture the noun phrase after a provenance verb.
SOURCE_CONTEXT_PATTERNS = [
    re.compile(r'(?:according\s+to|based\s+on|sourced?\s+from|resolved?\s+(?:by|via)|'
               r'determined\s+by|reported\s+by|per)\s+'
               r'([A-Z][\w\s]{0,30}?)(?:\.|,|;|\s+and\s|\s+or\s|\s+at\s|$)',
               re.IGNORECASE),
]


class DataSourceExtractor:
    """
    Extract data source from market text.
    
    Uses pattern matching for fast extraction.
    Searches both the statement and resolution criteria.
    """
    
    async def extract_data_source(
        self,
        resolution_criteria: Optional[str],
        statement: Optional[str] = None,
    ) -> Optional[str]:
        """
        Extract data source from resolution criteria and/or statement.
        
        Args:
            resolution_criteria: Resolution criteria text (primary).
            statement: Market statement text (fallback).
            
        Returns:
            Canonical data source name, or None if not detected.
        """
        # Combine both texts — criteria first (higher signal)
        parts = []
        if resolution_criteria:
            parts.append(resolution_criteria)
        if statement:
            parts.append(statement)
        if not parts:
            return None
        
        combined = " ".join(parts)
        combined_lower = combined.lower()
        
        # 1. Check known sources (exact pattern match)
        for source, patterns in KNOWN_SOURCES.items():
            for pattern in patterns:
                if re.search(pattern, combined_lower if pattern.islower() else combined):
                    return source
        
        # 2. Contextual extraction (provenance phrases)
        for pattern in SOURCE_CONTEXT_PATTERNS:
            match = pattern.search(combined)
            if match:
                candidate = match.group(1).strip()
                if candidate and len(candidate) > 1:
                    # Check if the captured phrase matches a known source
                    candidate_lower = candidate.lower()
                    for source, patterns in KNOWN_SOURCES.items():
                        for p in patterns:
                            if re.search(p, candidate_lower if p.islower() else candidate):
                                return source
                    # Return the raw captured name (title-cased) when
                    # it is not in our dictionary but was explicitly cited
                    return candidate.strip()
        
        return None

