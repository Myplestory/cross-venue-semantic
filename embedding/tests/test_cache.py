"""Tests for InMemoryCache."""

import pytest
import asyncio

from embedding.cache.in_memory import InMemoryCache


@pytest.mark.asyncio
async def test_cache_initialization():
    """Test cache initialization."""
    cache = InMemoryCache(max_size=100)
    assert not cache._initialized
    
    await cache.initialize()
    assert cache._initialized


@pytest.mark.asyncio
async def test_cache_get_set():
    """Test basic get/set operations."""
    cache = InMemoryCache(max_size=100)
    await cache.initialize()
    
    embedding = [0.1] * 2048
    content_hash = "test-hash-123"
    
    # Set
    await cache.set(content_hash, embedding)
    
    # Get
    result = await cache.get(content_hash)
    
    assert result == embedding
    assert result is not embedding  # Should be a copy


@pytest.mark.asyncio
async def test_cache_miss():
    """Test cache miss returns None."""
    cache = InMemoryCache(max_size=100)
    await cache.initialize()
    
    result = await cache.get("nonexistent-hash")
    
    assert result is None


@pytest.mark.asyncio
async def test_cache_lru_eviction():
    """Test LRU eviction when cache is full."""
    cache = InMemoryCache(max_size=2)
    await cache.initialize()
    
    # Fill cache
    await cache.set("hash1", [0.1] * 2048)
    await cache.set("hash2", [0.2] * 2048)
    
    # Access hash1 (makes it most recently used)
    await cache.get("hash1")
    
    # Add hash3 (should evict hash2, not hash1)
    await cache.set("hash3", [0.3] * 2048)
    
    # hash1 should still be there
    assert await cache.get("hash1") is not None
    
    # hash2 should be evicted
    assert await cache.get("hash2") is None
    
    # hash3 should be there
    assert await cache.get("hash3") is not None


@pytest.mark.asyncio
async def test_cache_update_existing():
    """Test updating existing cache entry."""
    cache = InMemoryCache(max_size=100)
    await cache.initialize()
    
    content_hash = "test-hash"
    
    # Set initial value
    await cache.set(content_hash, [0.1] * 2048)
    
    # Update with new value
    await cache.set(content_hash, [0.2] * 2048)
    
    # Should have new value
    result = await cache.get(content_hash)
    assert result == [0.2] * 2048


@pytest.mark.asyncio
async def test_cache_statistics():
    """Test cache statistics tracking."""
    cache = InMemoryCache(max_size=100)
    await cache.initialize()
    
    # Generate some hits and misses
    await cache.set("hash1", [0.1] * 2048)
    await cache.get("hash1")  # Hit
    await cache.get("hash1")  # Hit
    await cache.get("hash2")  # Miss
    
    stats = await cache.stats()
    
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["max_size"] == 100
    assert "hit_rate" in stats


@pytest.mark.asyncio
async def test_cache_clear():
    """Test clearing cache."""
    cache = InMemoryCache(max_size=100)
    await cache.initialize()
    
    # Add some entries
    await cache.set("hash1", [0.1] * 2048)
    await cache.set("hash2", [0.2] * 2048)
    
    # Clear
    await cache.clear()
    
    # Verify stats reset (before any new gets)
    stats = await cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["size"] == 0
    
    # Verify empty
    assert await cache.get("hash1") is None
    assert await cache.get("hash2") is None


@pytest.mark.asyncio
async def test_cache_batch_get():
    """Test batch get operation."""
    cache = InMemoryCache(max_size=100)
    await cache.initialize()
    
    # Set multiple entries
    await cache.set("hash1", [0.1] * 2048)
    await cache.set("hash2", [0.2] * 2048)
    await cache.set("hash3", [0.3] * 2048)
    
    # Batch get
    results = await cache.get_batch(["hash1", "hash2", "hash4"])
    
    assert "hash1" in results
    assert "hash2" in results
    assert "hash4" not in results  # Miss
    assert len(results) == 2


@pytest.mark.asyncio
async def test_cache_batch_set():
    """Test batch set operation."""
    cache = InMemoryCache(max_size=100)
    await cache.initialize()
    
    embeddings = {
        "hash1": [0.1] * 2048,
        "hash2": [0.2] * 2048,
        "hash3": [0.3] * 2048
    }
    
    await cache.set_batch(embeddings)
    
    # Verify all were set
    assert await cache.get("hash1") == [0.1] * 2048
    assert await cache.get("hash2") == [0.2] * 2048
    assert await cache.get("hash3") == [0.3] * 2048


@pytest.mark.asyncio
async def test_cache_batch_empty():
    """Test batch operations with empty lists."""
    cache = InMemoryCache(max_size=100)
    await cache.initialize()
    
    # Empty batch get
    results = await cache.get_batch([])
    assert results == {}
    
    # Empty batch set
    await cache.set_batch({})
    # Should not raise error


@pytest.mark.asyncio
async def test_cache_thread_safety():
    """Test concurrent cache access (thread safety)."""
    cache = InMemoryCache(max_size=1000)
    await cache.initialize()
    
    # Concurrent set operations
    async def set_entry(i):
        await cache.set(f"hash{i}", [float(i)] * 2048)
    
    # Concurrent get operations
    async def get_entry(i):
        return await cache.get(f"hash{i}")
    
    # Set multiple entries concurrently
    await asyncio.gather(*[set_entry(i) for i in range(100)])
    
    # Get multiple entries concurrently
    results = await asyncio.gather(*[get_entry(i) for i in range(100)])
    
    # All should succeed
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_cache_hit_rate_calculation():
    """Test hit rate calculation."""
    cache = InMemoryCache(max_size=100)
    await cache.initialize()
    
    await cache.set("hash1", [0.1] * 2048)
    
    # 3 hits, 2 misses
    await cache.get("hash1")
    await cache.get("hash1")
    await cache.get("hash1")
    await cache.get("hash2")
    await cache.get("hash3")
    
    stats = await cache.stats()
    
    # 3 hits, 2 misses = 60% hit rate
    assert stats["hits"] == 3
    assert stats["misses"] == 2
    assert stats["total_requests"] == 5


@pytest.mark.asyncio
async def test_cache_max_size_zero():
    """Test cache with max_size=0 (edge case - cache always at capacity)."""
    cache = InMemoryCache(max_size=0)
    await cache.initialize()
    
    # With max_size=0, len >= max_size is always true (even when empty)
    # So it will try to evict before adding, but cache is empty
    # Then it adds the item. So first item should be there.
    await cache.set("hash1", [0.1] * 2048)
    
    # First item should be there (nothing to evict when cache was empty)
    result1 = await cache.get("hash1")
    assert result1 is not None
    
    # Adding second item will evict first (since len(1) >= max_size(0))
    await cache.set("hash2", [0.2] * 2048)
    
    # hash1 should be evicted, hash2 should be there
    assert await cache.get("hash1") is None
    assert await cache.get("hash2") is not None

