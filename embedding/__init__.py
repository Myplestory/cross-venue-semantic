"""
Embedding module for semantic pipeline.

Converts canonicalized market text into vector embeddings
and stores them in Qdrant for similarity search.
"""

from .types import EmbeddedEvent
from .encoder import EmbeddingEncoder
from .index import QdrantIndex
from .processor import EmbeddingProcessor
from .cache.in_memory import InMemoryCache

__all__ = [
    "EmbeddedEvent",
    "EmbeddingEncoder",
    "QdrantIndex",
    "EmbeddingProcessor",
    "InMemoryCache",
]
