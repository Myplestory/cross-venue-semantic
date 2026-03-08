"""
Frontrunning opportunity detector.

Analyzes latency between game events and odds updates to identify
information advantage windows for potential frontrunning strategies.

Frontrunning detection is based on:
1. Latency between game events and odds updates (lower = better)
2. Price movement magnitude after events (higher = more significant)
3. Information advantage windows (time before market reacts)
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, UTC

logger = logging.getLogger(__name__)


class FrontrunningDetector:
    """
    Detects potential frontrunning opportunities by analyzing:
    1. Latency between game events and odds updates
    2. Price movement magnitude after events
    3. Information advantage windows
    
    A frontrunning opportunity exists when:
    - Game event occurs
    - Significant price movement follows within a short time window
    - Latency is low enough to execute before the market fully reacts
    """
    
    def __init__(
        self,
        min_price_move_pct: float = 0.02,  # 2% price move threshold
        max_latency_ms: float = 2000.0,  # 2 second window for frontrunning
        min_confidence: float = 0.6,  # Minimum confidence score to report
    ):
        """
        Initialize frontrunning detector.
        
        Args:
            min_price_move_pct: Minimum price movement percentage to consider
            max_latency_ms: Maximum latency for frontrunning window
            min_confidence: Minimum confidence score to report opportunity
        """
        self.min_price_move_pct = min_price_move_pct
        self.max_latency_ms = max_latency_ms
        self.min_confidence = min_confidence
    
    def analyze_opportunity(
        self,
        event: Dict[str, Any],
        price_before: float,
        price_after: float,
        latency_ms: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze if an event represents a frontrunning opportunity.
        
        Args:
            event: Game event dict with 'event_type', 'timestamp', etc.
            price_before: Market price before event (0.0 to 1.0)
            price_after: Market price after event (0.0 to 1.0)
            latency_ms: Latency in milliseconds between event and price update
        
        Returns:
            Dict with opportunity details if detected, None otherwise.
            Contains:
                - event_type: Type of game event
                - event_timestamp: When the event occurred
                - price_before: Price before event
                - price_after: Price after event
                - price_move_pct: Percentage price movement
                - latency_ms: Latency measurement
                - opportunity_type: "frontrunning"
                - confidence: Confidence score (0.0 to 1.0)
                - actionable: Whether opportunity is actionable (latency < threshold)
        """
        if price_before <= 0.0 or price_after <= 0.0:
            return None
        
        # Calculate price movement percentage
        price_move_pct = abs(price_after - price_before) / price_before
        
        # Check if meets minimum thresholds
        if latency_ms > self.max_latency_ms:
            return None  # Too slow to frontrun
        
        if price_move_pct < self.min_price_move_pct:
            return None  # Price movement too small
        
        # Calculate confidence score
        confidence = self._calculate_confidence(latency_ms, price_move_pct)
        
        if confidence < self.min_confidence:
            return None  # Confidence too low
        
        # Determine if actionable (can execute before market fully reacts)
        actionable = latency_ms < (self.max_latency_ms * 0.5)  # Within half of max window
        
        opportunity = {
            "event_type": event.get('event_type', 'unknown'),
            "event_timestamp": event.get('timestamp', datetime.now(UTC)),
            "price_before": price_before,
            "price_after": price_after,
            "price_move_pct": price_move_pct,
            "price_move_abs": abs(price_after - price_before),
            "latency_ms": latency_ms,
            "opportunity_type": "frontrunning",
            "confidence": confidence,
            "actionable": actionable,
            "source": event.get('source', 'unknown'),
        }
        
        logger.info(
            f"[FrontrunningDetector] Opportunity detected: "
            f"event={opportunity['event_type']}, "
            f"price_move={price_move_pct:.2%}, "
            f"latency={latency_ms:.1f}ms, "
            f"confidence={confidence:.2f}, "
            f"actionable={actionable}"
        )
        
        return opportunity
    
    def _calculate_confidence(
        self,
        latency_ms: float,
        price_move_pct: float
    ) -> float:
        """
        Calculate confidence score for frontrunning opportunity.
        
        Lower latency + higher price move = higher confidence.
        
        Args:
            latency_ms: Latency in milliseconds
            price_move_pct: Price movement percentage
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Latency score: lower latency = higher score
        # Normalize to 0-1 range (0ms = 1.0, max_latency_ms = 0.0)
        latency_score = max(0.0, 1.0 - (latency_ms / self.max_latency_ms))
        
        # Price move score: higher move = higher score
        # Normalize to 0-1 range (min_move = 0.0, 2x min_move = 1.0)
        move_score = min(1.0, price_move_pct / (self.min_price_move_pct * 2))
        
        # Weighted combination (latency more important for frontrunning)
        confidence = (latency_score * 0.6) + (move_score * 0.4)
        
        return confidence
    
    def should_alert(self, opportunity: Dict[str, Any]) -> bool:
        """
        Determine if an opportunity should trigger an alert.
        
        Args:
            opportunity: Opportunity dict from analyze_opportunity
        
        Returns:
            True if should alert, False otherwise
        """
        if not opportunity:
            return False
        
        # Alert if actionable and high confidence
        return (
            opportunity.get('actionable', False) and
            opportunity.get('confidence', 0.0) >= 0.7
        )

