"""
Embedding cache module.

Currently implements in-memory LRU cache.
Future: file-based, Redis, Qdrant cache implementations.
"""

from .in_memory import InMemoryCache

__all__ = ["InMemoryCache"]

