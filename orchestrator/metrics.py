"""
Pipeline metrics with thread-safe operations.

Fintech-grade metrics tracking with atomic operations.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

UTC = timezone.utc


@dataclass
class StageMetrics:
    """
    Per-stage latency and throughput tracker.
    
    Thread-safe: All mutations use asyncio.Lock for concurrent access.
    Designed for async coroutines running on the same event loop.
    """
    
    total_calls: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    
    # Thread-safety: Lock for concurrent access from multiple workers
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    
    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls
    
    async def record(self, latency_ms: float, *, error: bool = False) -> None:
        """
        Record a single stage execution.
        
        Thread-safe: Uses lock for concurrent access.
        
        Args:
            latency_ms: Execution latency in milliseconds
            error: Whether the execution resulted in an error
        """
        async with self._lock:
            self.total_calls += 1
            self.total_latency_ms += latency_ms
            if latency_ms > self.max_latency_ms:
                self.max_latency_ms = latency_ms
            if latency_ms < self.min_latency_ms:
                self.min_latency_ms = latency_ms
            if error:
                self.total_errors += 1
    
    def snapshot(self) -> Dict[str, Any]:
        """
        Return a JSON-safe snapshot.
        
        Thread-safe: Returns a copy of current state.
        """
        return {
            "calls": self.total_calls,
            "errors": self.total_errors,
            "avg_ms": round(self.avg_latency_ms, 1),
            "min_ms": round(self.min_latency_ms, 1) if self.total_calls else 0,
            "max_ms": round(self.max_latency_ms, 1),
        }


STAGE_NAMES = (
    "canonicalization",
    "embedding",
    "retrieval",
    "reranking",
    "extraction",
    "verification",
    "persistence",
)


@dataclass
class PipelineMetrics:
    """
    Aggregate pipeline metrics for health monitoring / Prometheus.
    
    Thread-safe: Uses locks for concurrent metric updates.
    """
    
    events_received: int = 0
    events_deduplicated: int = 0
    events_processed: int = 0
    events_failed: int = 0
    events_no_candidates: int = 0
    pairs_found: int = 0
    pairs_equivalent: int = 0
    pairs_needs_review: int = 0
    pairs_not_equivalent: int = 0
    pairs_persisted: int = 0
    started_at: Optional[datetime] = None
    
    # Per-stage metrics
    canonicalization: StageMetrics = field(default_factory=StageMetrics)
    embedding: StageMetrics = field(default_factory=StageMetrics)
    retrieval: StageMetrics = field(default_factory=StageMetrics)
    reranking: StageMetrics = field(default_factory=StageMetrics)
    extraction: StageMetrics = field(default_factory=StageMetrics)
    verification: StageMetrics = field(default_factory=StageMetrics)
    persistence: StageMetrics = field(default_factory=StageMetrics)
    
    # Thread-safety: Lock for counter updates
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    
    async def increment(self, counter: str, value: int = 1) -> None:
        """
        Thread-safe counter increment.
        
        Args:
            counter: Counter name (e.g., "events_received")
            value: Increment value (default: 1)
        """
        async with self._lock:
            if hasattr(self, counter):
                setattr(self, counter, getattr(self, counter) + value)
    
    def summary(self) -> Dict[str, Any]:
        """
        Return metrics snapshot for health endpoint / logging.
        
        Thread-safe: Returns a copy of current state.
        """
        uptime = 0.0
        if self.started_at:
            uptime = (datetime.now(UTC) - self.started_at).total_seconds()
        
        eps = (
            round(self.events_processed / (uptime / 60), 2)
            if uptime > 60
            else 0.0
        )
        
        return {
            "uptime_seconds": round(uptime, 1),
            "events": {
                "received": self.events_received,
                "deduplicated": self.events_deduplicated,
                "processed": self.events_processed,
                "failed": self.events_failed,
                "no_candidates": self.events_no_candidates,
            },
            "pairs": {
                "found": self.pairs_found,
                "equivalent": self.pairs_equivalent,
                "needs_review": self.pairs_needs_review,
                "not_equivalent": self.pairs_not_equivalent,
                "persisted": self.pairs_persisted,
            },
            "throughput_events_per_min": eps,
            "stages": {
                name: getattr(self, name).snapshot()
                for name in STAGE_NAMES
            },
        }

