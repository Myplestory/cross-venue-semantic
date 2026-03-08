"""
Real-time system health metrics and latency tracking.

Provides thread-safe metrics collection for monitoring system health,
performance, and compliance requirements.
"""

import asyncio
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional
from datetime import datetime, UTC

from .circuit_breaker import CircuitState


@dataclass
class LatencyMetrics:
    """
    SEC/FINRA-compliant latency tracking.
    
    Records the time delta between a game event and corresponding odds update.
    Used for frontrunning detection and information advantage analysis.
    """
    event_timestamp: datetime
    odds_update_timestamp: datetime
    latency_ms: float
    venue: str
    event_type: str
    market_id: str


@dataclass
class SystemMetrics:
    """
    Real-time system health metrics with thread-safe updates.
    
    Tracks:
    - WebSocket connection health (reconnects, errors)
    - API call success/failure rates
    - Orderbook update frequency
    - Game event processing
    - Arbitrage opportunity detection
    - Latency measurements (for compliance)
    - Circuit breaker states
    """
    websocket_reconnects: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    api_errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    api_successes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    orderbook_updates: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    game_events_received: int = 0
    arb_opportunities_detected: int = 0
    latency_samples: List[LatencyMetrics] = field(default_factory=list)
    circuit_breaker_state: Dict[str, str] = field(default_factory=dict)  # CircuitState enum values
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    async def record_latency(
        self,
        event_ts: datetime,
        odds_ts: datetime,
        venue: str,
        event_type: str,
        market_id: str
    ):
        """
        Record latency measurement for audit trail.
        
        Thread-safe operation for concurrent access.
        
        Args:
            event_ts: Game event timestamp
            odds_ts: Odds update timestamp
            venue: Venue name (e.g., "kalshi", "polymarket")
            event_type: Type of game event (e.g., "CHAMPION_KILL", "BUILDING_KILL")
            market_id: Market identifier
        """
        latency_ms = (odds_ts - event_ts).total_seconds() * 1000
        
        async with self._lock:
            self.latency_samples.append(LatencyMetrics(
                event_timestamp=event_ts,
                odds_update_timestamp=odds_ts,
                latency_ms=latency_ms,
                venue=venue,
                event_type=event_type,
                market_id=market_id,
            ))
            # Keep only last 1000 samples for memory efficiency
            if len(self.latency_samples) > 1000:
                self.latency_samples = self.latency_samples[-1000:]
    
    async def increment_websocket_reconnect(self, feed_name: str):
        """Increment WebSocket reconnect counter (thread-safe)."""
        async with self._lock:
            self.websocket_reconnects[feed_name] += 1
    
    async def increment_api_error(self, api_name: str):
        """Increment API error counter (thread-safe)."""
        async with self._lock:
            self.api_errors[api_name] += 1
    
    async def increment_api_success(self, api_name: str):
        """Increment API success counter (thread-safe)."""
        async with self._lock:
            self.api_successes[api_name] += 1
    
    async def increment_orderbook_update(self, venue: str):
        """Increment orderbook update counter (thread-safe)."""
        async with self._lock:
            self.orderbook_updates[venue] += 1
    
    async def increment_game_events(self, count: int = 1):
        """Increment game events received counter (thread-safe)."""
        async with self._lock:
            self.game_events_received += count
    
    async def increment_arb_opportunities(self, count: int = 1):
        """Increment arbitrage opportunities detected counter (thread-safe)."""
        async with self._lock:
            self.arb_opportunities_detected += count
    
    async def update_circuit_breaker_state(self, circuit_name: str, state: CircuitState):
        """
        Update circuit breaker state (thread-safe).
        
        Args:
            circuit_name: Name/identifier of the circuit breaker
            state: Current CircuitState
        """
        async with self._lock:
            self.circuit_breaker_state[circuit_name] = state.value
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency statistics from recent samples.
        
        Returns:
            Dict with min, max, mean, median latency in milliseconds
        """
        if not self.latency_samples:
            return {
                "count": 0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "mean_ms": 0.0,
                "median_ms": 0.0,
            }
        
        latencies = [sample.latency_ms for sample in self.latency_samples]
        sorted_latencies = sorted(latencies)
        count = len(sorted_latencies)
        
        return {
            "count": count,
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "mean_ms": sum(latencies) / count,
            "median_ms": sorted_latencies[count // 2] if count > 0 else 0.0,
        }
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dict with all metrics for reporting/monitoring
        """
        return {
            "websocket_reconnects": dict(self.websocket_reconnects),
            "api_errors": dict(self.api_errors),
            "api_successes": dict(self.api_successes),
            "api_error_rates": {
                name: errors / (errors + self.api_successes.get(name, 0))
                if (errors + self.api_successes.get(name, 0)) > 0 else 0.0
                for name, errors in self.api_errors.items()
            },
            "orderbook_updates": dict(self.orderbook_updates),
            "game_events_received": self.game_events_received,
            "arb_opportunities_detected": self.arb_opportunities_detected,
            "latency_stats": self.get_latency_stats(),
            "circuit_breaker_states": dict(self.circuit_breaker_state),
        }

