"""
Semantic Pipeline - Market Discovery and Matching

Separate Python process for:
- Market discovery from multiple venues (Kalshi, Polymarket, etc.)
- Canonical text normalization and hashing
- Embedding generation and vector search
- Cross-encoder verification
- LLM-based contract spec extraction and pair verification
- Writing verified pairs to database
"""

# Load configuration early (loads .env file)
from . import config  # noqa: F401

__version__ = "0.1.0"
