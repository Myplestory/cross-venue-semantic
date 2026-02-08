"""
Deduplication module for market events.

Hash-based deduplication using identity hash (venue:market_id).
Prevents processing duplicate market events.
"""

import hashlib
import logging
from typing import Set, Optional
from datetime import datetime, timedelta

from .types import MarketEvent, VenueType


logger = logging.getLogger(__name__)


class MarketDeduplicator:
    """
    Deduplicates market events using identity hash.
    
    Uses in-memory set with optional TTL for memory efficiency.
    """
    
    def __init__(self, ttl_seconds: Optional[int] = 3600):
        """
        Initialize deduplicator.
        
        Args:
            ttl_seconds: Time-to-live for seen hashes (None = no expiration)
        """
        self.ttl_seconds = ttl_seconds
        self._seen: Set[str] = set()
        self._seen_timestamps: dict[str, datetime] = {}
    
    def _identity_hash(self, event: MarketEvent) -> str:
        """
        Generate identity hash for deduplication.
        
        Format: venue:market_id
        """
        key = f"{event.venue.value}:{event.venue_market_id}"
        return hashlib.sha256(key.encode()).hexdigest()
    
    def is_duplicate(self, event: MarketEvent) -> bool:
        """
        Check if event is a duplicate.
        
        Args:
            event: Market event to check
            
        Returns:
            True if duplicate, False otherwise
        """
        identity_hash = self._identity_hash(event)
        
        # Cleanup expired entries
        if self.ttl_seconds:
            self._cleanup_expired()
        
        # Check if seen
        if identity_hash in self._seen:
            return True
        
        # Mark as seen
        self._seen.add(identity_hash)
        if self.ttl_seconds:
            self._seen_timestamps[identity_hash] = datetime.utcnow()
        
        return False
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from seen set."""
        if not self.ttl_seconds:
            return
        
        now = datetime.utcnow()
        expired = [
            hash_val
            for hash_val, timestamp in self._seen_timestamps.items()
            if (now - timestamp).total_seconds() > self.ttl_seconds
        ]
        
        for hash_val in expired:
            self._seen.discard(hash_val)
            self._seen_timestamps.pop(hash_val, None)
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired dedup entries")
    
    def clear(self) -> None:
        """Clear all seen entries."""
        self._seen.clear()
        self._seen_timestamps.clear()
    
    def size(self) -> int:
        """Get number of seen entries."""
        return len(self._seen)
