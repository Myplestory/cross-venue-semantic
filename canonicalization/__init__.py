"""
Canonicalization Module

Normalizes venue-specific market data into canonical format.
Handles text building, hashing, and ContractSpec models.
"""

from .text_builder import (
    CanonicalTextBuilder,
    KalshiTextBuilder,
    PolymarketTextBuilder,
    get_builder,
)
from .hasher import ContentHasher
from .contract_spec import (
    ContractSpec,
    DateSpec,
    EntitySpec,
    ThresholdSpec,
)
from .types import CanonicalEvent

__all__ = [
    "CanonicalTextBuilder",
    "KalshiTextBuilder",
    "PolymarketTextBuilder",
    "get_builder",
    "ContentHasher",
    "ContractSpec",
    "DateSpec",
    "EntitySpec",
    "ThresholdSpec",
    "CanonicalEvent",
]
