"""
Core monitoring logic for esports arbitrage detection.

This module provides:
- Orderbook management for single market pairs
- Latency correlation between game events and odds updates
- Frontrunning opportunity detection
"""

from .orderbook_manager import SingleMarketOrderbookManager
from .latency_engine import LatencyCorrelationEngine
from .frontrunning_detector import FrontrunningDetector

# Re-export LatencyMetrics from compliance module for convenience
from ..compliance.metrics import LatencyMetrics

__all__ = [
    "SingleMarketOrderbookManager",
    "LatencyCorrelationEngine",
    "LatencyMetrics",
    "FrontrunningDetector",
]

