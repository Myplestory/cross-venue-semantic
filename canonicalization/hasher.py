"""
Hashing utilities for market deduplication and change detection.

Two hash strategies:
1. Identity hash: venue:market_id (for deduplication)
2. Content hash: normalized canonical text (for change detection)
"""

import hashlib
import asyncio
import logging
from typing import List, Optional
from discovery.types import VenueType

logger = logging.getLogger(__name__)


class ContentHasher:
    """
    Generate stable content hash from canonical text.
    
    All operations are CPU-bound but fast (<1ms), so synchronous is acceptable.
    Wrapped in async functions for pipeline consistency.
    """
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for stable hashing (synchronous, fast).
        
        Normalization rules:
        1. Normalize line endings (\r\n, \r → \n)
        2. Strip trailing whitespace per line
        3. Collapse multiple blank lines to single blank line
        4. Remove trailing blank lines
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # 1. Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 2. Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.split('\n')]
        
        # 3. Remove trailing blank lines
        while lines and not lines[-1]:
            lines.pop()
        
        # 4. Collapse multiple blank lines to single blank line
        normalized = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue  # Skip duplicate blank line
            normalized.append(line)
            prev_blank = is_blank
        
        return '\n'.join(normalized)
    
    @staticmethod
    def hash_content(text: str) -> str:
        """
        Generate SHA-256 hash of normalized text (synchronous, fast).
        
        Args:
            text: Canonical text to hash
            
        Returns:
            SHA-256 hex digest
        """
        normalized = ContentHasher.normalize_text(text)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    @staticmethod
    async def hash_content_async(text: str) -> str:
        """
        Async wrapper for hash_content().
        
        For consistency with async pipeline, though operation is fast enough
        to be synchronous.
        
        Args:
            text: Canonical text to hash
            
        Returns:
            SHA-256 hex digest
        """
        return ContentHasher.hash_content(text)
    
    @staticmethod
    async def hash_batch(texts: List[str]) -> List[str]:
        """
        Hash multiple texts in parallel (non-blocking).
        
        Uses asyncio.gather() for parallelization.
        
        Args:
            texts: List of canonical texts to hash
            
        Returns:
            List of SHA-256 hex digests (same order as input)
        """
        tasks = [ContentHasher.hash_content_async(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        hashes = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error hashing text {i}: {result}")
                hashes.append("")  # Empty hash for failed items
            else:
                hashes.append(result)
        
        return hashes
    
    @staticmethod
    def identity_hash(venue: VenueType, market_id: str) -> str:
        """
        Generate identity hash (synchronous, very fast).
        
        Format: sha256(f"{venue}:{market_id}")
        
        Already implemented in discovery/dedup.py, but included here
        for completeness and consistency.
        
        Args:
            venue: Venue type
            market_id: Market ID from venue
            
        Returns:
            SHA-256 hex digest
        """
        key = f"{venue.value}:{market_id}"
        return hashlib.sha256(key.encode('utf-8')).hexdigest()
