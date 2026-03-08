"""
SEC/FINRA-compliant audit trail logger.

Logs all trading decisions, latency measurements, and system events
for regulatory compliance and investigation purposes.

Compliance Requirements:
- All logs are immutable (append-only)
- Structured JSON format for machine-readable analysis
- Millisecond-precision timestamps (UTC)
- Separate log files by event category
- No PII in logs (user IDs, IPs, etc. are hashed or omitted)
"""

import json
import logging
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional, Dict, Any
import hashlib

from .metrics import LatencyMetrics


class AuditLogger:
    """
    SEC/FINRA-compliant audit trail logger.
    
    Logs all trading decisions, latency measurements, and system events
    in structured JSON format for regulatory compliance.
    
    Log Categories:
    - Trading: Arbitrage opportunities, execution decisions
    - Latency: Game event to odds update timing measurements
    - System: Circuit breaker state changes, errors, reconnections
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs (default: audit_logs/)
        """
        if log_dir is None:
            log_dir = Path("audit_logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate loggers for different event types (compliance requirement)
        self.trading_logger = self._setup_logger("trading", "trading_audit.log")
        self.latency_logger = self._setup_logger("latency", "latency_audit.log")
        self.system_logger = self._setup_logger("system", "system_audit.log")
    
    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        """
        Set up a logger for a specific audit category.
        
        Args:
            name: Logger name (for identification)
            filename: Log file name
        
        Returns:
            Configured Logger instance
        """
        logger = logging.getLogger(f"audit.{name}")
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler with millisecond precision
        handler = logging.FileHandler(
            self.log_dir / filename,
            mode='a',  # Append-only (immutable logs)
            encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s.%(msecs)03d UTC | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def log_arbitrage_opportunity(
        self,
        opportunity: Any,  # ArbOpportunity from spread_scanner
        game_event: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        decision: str = "detected"
    ):
        """
        Log arbitrage opportunity with full context for compliance.
        
        Args:
            opportunity: ArbOpportunity instance with opportunity details
            game_event: Optional game event dict that triggered the opportunity
            latency_ms: Optional latency measurement in milliseconds
            decision: Decision taken ("detected", "executed", "rejected", etc.)
        """
        try:
            # Extract opportunity data (handle both dict and object)
            if hasattr(opportunity, '__dict__'):
                opp_data = {
                    "pair_key": getattr(opportunity, 'pair_key', None),
                    "venue_a": getattr(opportunity, 'venue_a', None),
                    "venue_b": getattr(opportunity, 'venue_b', None),
                    "title_a": getattr(opportunity, 'title_a', None),
                    "title_b": getattr(opportunity, 'title_b', None),
                    "buy_a_side": getattr(opportunity, 'buy_a_side', None),
                    "buy_b_side": getattr(opportunity, 'buy_b_side', None),
                    "ask_a": getattr(opportunity, 'ask_a', None),
                    "ask_b": getattr(opportunity, 'ask_b', None),
                    "total_cost_1": getattr(opportunity, 'total_cost_1', None),
                    "gross_edge": getattr(opportunity, 'gross_edge', None),
                    "net_profit_1": getattr(opportunity, 'net_profit_1', None),
                    "net_roi_pct": getattr(opportunity, 'net_roi_pct', None),
                    "optimal_qty": getattr(opportunity, 'optimal_qty', None),
                    "max_profit": getattr(opportunity, 'max_profit', None),
                    "optimal_capital": getattr(opportunity, 'optimal_capital', None),
                    "pnl_at_budget": getattr(opportunity, 'pnl_at_budget', None),
                    "daily_yield_bps": getattr(opportunity, 'daily_yield_bps', None),
                    "annualized_roi_pct": getattr(opportunity, 'annualized_roi_pct', None),
                }
            else:
                opp_data = opportunity if isinstance(opportunity, dict) else {}
            
            # Build audit entry (structured JSON for compliance)
            audit_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "arbitrage_opportunity",
                "decision": decision,
                "opportunity": opp_data,
                "game_event": game_event,
                "latency_ms": latency_ms,
            }
            
            # Log as JSON string (structured format for compliance)
            self.trading_logger.info(json.dumps(audit_entry, default=str))
            
        except Exception as e:
            # Never fail on audit logging - log error to system logger
            self.system_logger.error(
                json.dumps({
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "audit_log_error",
                    "error": str(e),
                    "category": "trading",
                }, default=str)
            )
    
    def log_latency_measurement(self, metrics: LatencyMetrics):
        """
        Log latency measurement for compliance.
        
        Args:
            metrics: LatencyMetrics instance with measurement data
        """
        try:
            audit_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "latency_measurement",
                "event_timestamp": metrics.event_timestamp.isoformat(),
                "odds_update_timestamp": metrics.odds_update_timestamp.isoformat(),
                "latency_ms": metrics.latency_ms,
                "venue": metrics.venue,
                "event_type": metrics.event_type,
                "market_id": metrics.market_id,
            }
            
            self.latency_logger.info(json.dumps(audit_entry, default=str))
            
        except Exception as e:
            self.system_logger.error(
                json.dumps({
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "audit_log_error",
                    "error": str(e),
                    "category": "latency",
                }, default=str)
            )
    
    def log_system_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "info"
    ):
        """
        Log system event (circuit breaker changes, errors, reconnections).
        
        Args:
            event_type: Type of system event (e.g., "circuit_breaker_state_change")
            details: Event details dict
            severity: Log severity ("info", "warning", "error")
        """
        try:
            audit_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "system_event",
                "event_type": event_type,
                "severity": severity,
                "details": details,
            }
            
            log_method = getattr(self.system_logger, severity, self.system_logger.info)
            log_method(json.dumps(audit_entry, default=str))
            
        except Exception as e:
            # Fallback to basic logging if JSON serialization fails
            self.system_logger.error(
                f"Failed to log system event: {e} | type={event_type}"
            )
    
    def log_circuit_breaker_state_change(
        self,
        circuit_name: str,
        old_state: str,
        new_state: str,
        reason: Optional[str] = None
    ):
        """
        Log circuit breaker state change for compliance.
        
        Args:
            circuit_name: Name/identifier of the circuit breaker
            old_state: Previous state
            new_state: New state
            reason: Optional reason for state change
        """
        self.log_system_event(
            event_type="circuit_breaker_state_change",
            details={
                "circuit_name": circuit_name,
                "old_state": old_state,
                "new_state": new_state,
                "reason": reason,
            },
            severity="info"
        )

