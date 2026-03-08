"""
Fintech-grade circuit breaker for API resilience.

Prevents cascading failures and protects downstream systems by
automatically opening circuits when failure thresholds are exceeded.

Implements industry-standard circuit breaker pattern with:
- Three-state machine (CLOSED, OPEN, HALF_OPEN)
- Configurable failure thresholds and recovery timeouts
- Thread-safe async operations
- Comprehensive logging for compliance
"""

import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from datetime import datetime, UTC

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state."""
    CLOSED = "closed"      # Normal operation, allow requests
    OPEN = "open"          # Failing, reject requests immediately
    HALF_OPEN = "half_open"  # Testing recovery, allow limited requests


class CircuitBreakerOpenError(Exception):
    """
    Raised when circuit breaker is OPEN.
    
    This exception indicates the circuit is in an OPEN state and
    requests are being rejected to prevent cascading failures.
    """
    pass


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.
    
    Attributes:
        failure_threshold: Number of consecutive failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        success_threshold: Consecutive successes needed to close from HALF_OPEN
        timeout: Maximum time to wait for function execution (None = no timeout)
        name: Optional name for logging/identification
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    timeout: Optional[float] = None  # None = no timeout
    name: Optional[str] = None


class CircuitBreaker:
    """
    Fintech-grade circuit breaker for API resilience.
    
    Prevents cascading failures and protects downstream systems by
    automatically opening circuits when failure thresholds are exceeded.
    
    States:
    - CLOSED: Normal operation, allow all requests
    - OPEN: Service failing, reject requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests
    
    Transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN → CLOSED: After success_threshold consecutive successes
    - HALF_OPEN → OPEN: After any failure
    
    Thread-safe: All state changes are protected by async locks.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration (uses defaults if None)
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._name = self.config.name or "CircuitBreaker"
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            Result from function execution
        
        Raises:
            CircuitBreakerOpenError: If circuit is OPEN and recovery not ready
            asyncio.TimeoutError: If function execution exceeds timeout
            Exception: Any exception raised by the function
        """
        # Check state and attempt recovery if needed
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    logger.info(
                        f"[{self._name}] Attempting recovery (OPEN → HALF_OPEN). "
                        f"Last failure: {self.last_failure_time}"
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    elapsed = (
                        (datetime.now(UTC) - self.last_failure_time).total_seconds()
                        if self.last_failure_time else 0
                    )
                    remaining = max(0, self.config.recovery_timeout - elapsed)
                    raise CircuitBreakerOpenError(
                        f"[{self._name}] Circuit breaker is OPEN. "
                        f"Last failure: {self.last_failure_time}. "
                        f"Retry after {remaining:.1f}s (recovery_timeout={self.config.recovery_timeout}s)"
                    )
        
        # Execute function with optional timeout
        try:
            if self.config.timeout is not None:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await func(*args, **kwargs)
            
            await self._record_success()
            return result
            
        except asyncio.TimeoutError as e:
            await self._record_failure()
            logger.warning(
                f"[{self._name}] Function call timeout after {self.config.timeout}s"
            )
            raise RuntimeError(
                f"[{self._name}] Function call timeout: {e}"
            ) from e
        except Exception as e:
            await self._record_failure()
            # Re-raise original exception (don't wrap)
            raise
    
    async def _record_success(self):
        """Record successful call and update state accordingly."""
        async with self._lock:
            self.last_success_time = datetime.now(UTC)
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info(
                        f"[{self._name}] Recovery successful (HALF_OPEN → CLOSED). "
                        f"Success count: {self.success_count}/{self.config.success_threshold}"
                    )
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success in CLOSED state
                if self.failure_count > 0:
                    logger.debug(
                        f"[{self._name}] Success in CLOSED state, resetting failure count "
                        f"(was {self.failure_count})"
                    )
                    self.failure_count = 0
    
    async def _record_failure(self):
        """Record failed call and update state accordingly."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(UTC)
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"[{self._name}] Recovery failed (HALF_OPEN → OPEN). "
                    f"Success count was: {self.success_count}/{self.config.success_threshold}"
                )
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    logger.error(
                        f"[{self._name}] Opening circuit (CLOSED → OPEN) after "
                        f"{self.failure_count} consecutive failures "
                        f"(threshold={self.config.failure_threshold})"
                    )
                    self.state = CircuitState.OPEN
    
    def _should_attempt_recovery(self) -> bool:
        """
        Check if enough time has passed to attempt recovery.
        
        Returns:
            True if recovery_timeout has elapsed since last failure
        """
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now(UTC) - self.last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout
    
    def is_open(self) -> bool:
        """
        Check if circuit is currently open.
        
        Returns:
            True if state is OPEN
        """
        return self.state == CircuitState.OPEN
    
    def get_state(self) -> CircuitState:
        """
        Get current circuit state.
        
        Returns:
            Current CircuitState enum value
        """
        return self.state
    
    def get_stats(self) -> dict:
        """
        Get current circuit breaker statistics.
        
        Returns:
            Dict with state, counts, and timestamps
        """
        return {
            "name": self._name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }

