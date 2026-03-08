"""
Latency correlation engine for measuring time between game events and odds updates.

Detects information advantage windows and frontrunning opportunities by correlating
game state changes with market price movements.
"""

import asyncio
import logging
from collections import deque
from typing import Deque, Optional, Dict, Any
from datetime import datetime, UTC

import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spread_scanner import VenueBook
from ..compliance.metrics import LatencyMetrics, SystemMetrics
from ..compliance.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class LatencyCorrelationEngine:
    """
    Correlates game events with odds updates to measure latency.
    
    Detects information advantage (frontrunning opportunities) by tracking
    the time between in-game events and corresponding market price movements.
    
    Uses a sliding window approach to correlate events with orderbook updates
    within a configurable time window (default 5 seconds).
    """
    
    def __init__(
        self,
        correlation_window_ms: float = 5000.0,  # 5 second window
        metrics: Optional[SystemMetrics] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """
        Initialize latency correlation engine.
        
        Args:
            correlation_window_ms: Time window in milliseconds for correlation
            metrics: SystemMetrics instance for recording measurements
            audit_logger: AuditLogger instance for compliance logging
        """
        self.correlation_window_ms = correlation_window_ms
        self.metrics = metrics
        self.audit_logger = audit_logger
        self.pending_events: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.orderbook_snapshots: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._lock = asyncio.Lock()
    
    async def record_game_event(self, event: Dict[str, Any]):
        """
        Record a game event and attempt correlation with recent odds updates.
        
        Args:
            event: Game event dict with:
                - 'timestamp': datetime of event
                - 'event_type': Type of event (e.g., 'CHAMPION_KILL', 'BUILDING_KILL')
                - 'source': Source of event (e.g., 'riot_api', 'polymarket_sports')
                - 'event_data': Additional event data
        """
        if 'timestamp' not in event:
            event['timestamp'] = datetime.now(UTC)
        
        async with self._lock:
            self.pending_events.append(event)
            
            if self.metrics:
                await self.metrics.increment_game_events()
        
        # Attempt correlation with recent orderbook updates
        await self._attempt_correlation()
    
    async def record_orderbook_update(
        self,
        venue: str,
        market_id: str,
        book: VenueBook
    ):
        """
        Record orderbook update and attempt correlation with recent game events.
        
        Args:
            venue: Venue name (e.g., 'kalshi', 'polymarket')
            market_id: Market identifier
            book: VenueBook instance with current prices
        """
        snapshot = {
            "timestamp": datetime.now(UTC),
            "venue": venue,
            "market_id": market_id,
            "yes_price": book.yes_ask_top if book.yes_asks else None,
            "no_price": book.no_ask_top if book.no_asks else None,
            "yes_depth": book.yes_depth if book.yes_asks else 0.0,
            "no_depth": book.no_depth if book.no_asks else 0.0,
        }
        
        async with self._lock:
            self.orderbook_snapshots.append(snapshot)
            
            if self.metrics:
                await self.metrics.increment_orderbook_update(venue)
        
        # Attempt correlation with recent game events
        await self._attempt_correlation()
    
    async def _attempt_correlation(self):
        """
        Correlate game events with orderbook updates within time window.
        
        For each pending event, find orderbook snapshots within the correlation
        window and record latency measurements.
        """
        async with self._lock:
            # Process events in chronological order
            events_to_process = list(self.pending_events)
            snapshots_to_check = list(self.orderbook_snapshots)
        
        correlated_events = []
        
        for event in events_to_process:
            event_ts = event.get('timestamp')
            if not isinstance(event_ts, datetime):
                continue
            
            event_type = event.get('event_type', 'unknown')
            
            # Find snapshots within correlation window
            for snapshot in snapshots_to_check:
                snapshot_ts = snapshot.get('timestamp')
                if not isinstance(snapshot_ts, datetime):
                    continue
                
                # Calculate latency (snapshot time - event time)
                latency_ms = (snapshot_ts - event_ts).total_seconds() * 1000
                
                # Check if within correlation window
                if 0 <= latency_ms <= self.correlation_window_ms:
                    # Record latency measurement
                    metrics = LatencyMetrics(
                        event_timestamp=event_ts,
                        odds_update_timestamp=snapshot_ts,
                        latency_ms=latency_ms,
                        venue=snapshot['venue'],
                        event_type=event_type,
                        market_id=snapshot['market_id'],
                    )
                    
                    # Record in system metrics
                    if self.metrics:
                        await self.metrics.record_latency(
                            event_ts,
                            snapshot_ts,
                            snapshot['venue'],
                            event_type,
                            snapshot['market_id'],
                        )
                    
                    # Log for audit trail
                    if self.audit_logger:
                        self.audit_logger.log_latency_measurement(metrics)
                    
                    logger.debug(
                        f"[LatencyEngine] Correlated event: {event_type} -> "
                        f"{snapshot['venue']} odds update, latency={latency_ms:.1f}ms"
                    )
                    
                    # Mark event as correlated (remove from pending)
                    correlated_events.append(event)
                    break
        
        # Remove correlated events from pending list
        if correlated_events:
            async with self._lock:
                for event in correlated_events:
                    if event in self.pending_events:
                        self.pending_events.remove(event)
        
        # Clean up old snapshots (outside correlation window)
        await self._cleanup_old_snapshots()
    
    async def _cleanup_old_snapshots(self):
        """Remove orderbook snapshots that are too old to correlate."""
        now = datetime.now(UTC)
        cutoff = now.timestamp() * 1000 - self.correlation_window_ms
        
        async with self._lock:
            # Keep only snapshots within correlation window
            valid_snapshots = [
                s for s in self.orderbook_snapshots
                if isinstance(s.get('timestamp'), datetime)
                and s['timestamp'].timestamp() * 1000 >= cutoff
            ]
            self.orderbook_snapshots.clear()
            self.orderbook_snapshots.extend(valid_snapshots)
    
    def get_pending_events_count(self) -> int:
        """Get number of pending events awaiting correlation."""
        return len(self.pending_events)
    
    def get_snapshot_count(self) -> int:
        """Get number of orderbook snapshots available for correlation."""
        return len(self.orderbook_snapshots)

