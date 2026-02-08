"""
Data types for canonicalization module.
"""

from dataclasses import dataclass
from datetime import datetime
from discovery.types import MarketEvent


@dataclass
class CanonicalEvent:
    """
    Canonicalized market event with hashes.
    
    Output of canonicalization phase, input to embedding phase.
    """
    event: MarketEvent
    canonical_text: str
    content_hash: str  # For change detection
    identity_hash: str  # For deduplication (from discovery)
    created_at: datetime = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()

