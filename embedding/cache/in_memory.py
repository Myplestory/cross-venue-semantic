"""
In-memory LRU cache for embeddings.

Simple, fast, no external dependencies.
Cache is lost on process restart.
"""

import asyncio
import logging
from typing import List, Optional, Dict
from collections import OrderedDict

logger = logging.getLogger(__name__)


class InMemoryCache:
    """
    In-memory LRU cache implementation for embedding vectors.
    
    Uses OrderedDict for LRU eviction. Thread-safe with asyncio.Lock.
    Fast lookups (<1ms) but cache is lost on process restart.
    
    Features:
    - LRU eviction when cache reaches max_size
    - Thread-safe async operations
    - Cache statistics (hits, misses, hit rate)
    - Batch operations for efficiency
    """
    
    def __init__(
        self,
        max_size: int = 10000,
    ):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of embeddings to cache (LRU eviction)
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._initialized = False
        
        logger.info(
            f"InMemoryCache initialized: max_size={max_size}"
        )
    
    async def initialize(self) -> None:
        """
        Initialize cache (no-op for in-memory, but required for consistency).
        
        Called once before use. For in-memory cache, this is a no-op,
        but it's included for consistency with future cache implementations
        that may need async initialization (e.g., file loading, connection setup).
        """
        if self._initialized:
            return
        
        self._initialized = True
        logger.debug("InMemoryCache initialized")
    
    async def get(self, content_hash: str) -> Optional[List[float]]:
        """
        Retrieve cached embedding (LRU: move to end).
        
        Args:
            content_hash: Content hash from canonicalization phase
            
        Returns:
            Embedding vector if found, None otherwise
        """
        async with self._lock:
            if content_hash in self._cache:
                # Move to end (most recently used)
                embedding = self._cache.pop(content_hash)
                self._cache[content_hash] = embedding
                self._hits += 1
                return embedding.copy()  # Return copy to prevent mutation
        
        self._misses += 1
        return None
    
    async def set(
        self,
        content_hash: str,
        embedding: List[float]
    ) -> None:
        """
        Cache embedding (LRU: evict oldest if at capacity).
        
        Args:
            content_hash: Content hash from canonicalization phase
            embedding: Embedding vector to cache
        """
        async with self._lock:
            # Remove if exists (update existing entry)
            if content_hash in self._cache:
                del self._cache[content_hash]
            
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size and len(self._cache) > 0:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"LRU eviction: removed {oldest_key[:8]}...")
            
            # Add to end (most recently used)
            self._cache[content_hash] = embedding.copy()  # Store copy to prevent mutation
    
    async def get_batch(
        self,
        content_hashes: List[str]
    ) -> Dict[str, List[float]]:
        """
        Batch retrieval for efficiency.
        
        Args:
            content_hashes: List of content hashes
            
        Returns:
            Dictionary mapping content_hash → embedding
        """
        if not content_hashes:
            return {}
        
        # Retrieve all in parallel
        tasks = [self.get(ch) for ch in content_hashes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary (only include successful retrievals)
        batch_results = {}
        for content_hash, result in zip(content_hashes, results):
            if not isinstance(result, Exception) and result is not None:
                batch_results[content_hash] = result
        
        return batch_results
    
    async def set_batch(
        self,
        embeddings: Dict[str, List[float]]
    ) -> None:
        """
        Batch storage for efficiency.
        
        Args:
            embeddings: Dictionary mapping content_hash → embedding
        """
        if not embeddings:
            return
        
        # Store all in parallel
        tasks = [
            self.set(content_hash, embedding)
            for content_hash, embedding in embeddings.items()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def clear(self) -> None:
        """
        Clear all cached entries.
        
        Resets cache and statistics.
        """
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("InMemoryCache cleared")
    
    async def stats(self) -> Dict[str, any]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache statistics:
            - type: Cache type identifier
            - size: Current number of cached entries
            - max_size: Maximum cache size
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Cache hit rate percentage
            - total_requests: Total get() requests
        """
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                "type": "InMemoryCache",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2f}%",
                "total_requests": total_requests,
            }

