"""
Data types for embedding module.
"""

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import List

# Import from parent module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonicalization.types import CanonicalEvent


@dataclass
class EmbeddedEvent:
    """
    Embedded canonical event ready for vector storage.
    
    Output of embedding phase, input to matching phase.
    """
    canonical_event: CanonicalEvent
    embedding: List[float]  # Vector embedding (2048-dim default for 4B model)
    embedding_model: str  # Model identifier (e.g., "Qwen/Qwen3-Embedding-4B")
    embedding_dim: int  # Vector dimension (2048 default, configurable 32-2560)
    created_at: datetime = None
    
    def __post_init__(self):
        """Initialize defaults and validate embedding dimension."""
        if self.created_at is None:
            self.created_at = datetime.now(UTC)
        
        # Make a copy of the embedding list to ensure immutability
        self.embedding = list(self.embedding)
        
        # Validate embedding dimension matches expected size
        if len(self.embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.embedding_dim}, got {len(self.embedding)}"
            )

