"""
Circuit breaker pattern for LLM fallback.

Prevents cascade failures when LLM API is down or rate-limited.
"""

import asyncio
import logging
from enum import Enum
from typing import Optional, Callable, Any
from datetime import datetime, UTC
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = config.EXTRACTION_CIRCUIT_BREAKER_FAILURE_THRESHOLD
    recovery_timeout: float = config.EXTRACTION_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
    success_threshold: int = config.EXTRACTION_CIRCUIT_BREAKER_SUCCESS_THRESHOLD
    timeout: float = config.EXTRACTION_CIRCUIT_BREAKER_TIMEOUT


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting LLM fallback.
    
    States:
    - CLOSED: Normal operation, allow requests
    - OPEN: Service failing, reject requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests
    
    Transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN → CLOSED: After success_threshold consecutive successes
    - HALF_OPEN → OPEN: After any failure
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result from function
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: If function call fails
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    logger.info("Circuit breaker: Attempting recovery (OPEN → HALF_OPEN)")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. "
                        f"Last failure: {self.last_failure_time}. "
                        f"Retry after {self.config.recovery_timeout}s"
                    )
        
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            await self._record_success()
            return result
            
        except asyncio.TimeoutError as e:
            await self._record_failure()
            raise RuntimeError(f"LLM call timeout: {e}") from e
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self):
        """Record successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info("Circuit breaker: Recovery successful (HALF_OPEN → CLOSED)")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    async def _record_failure(self):
        """Record failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(UTC)
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker: Recovery failed (HALF_OPEN → OPEN)")
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    logger.error(
                        f"Circuit breaker: Opening circuit after {self.failure_count} failures"
                    )
                    self.state = CircuitState.OPEN
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now(UTC) - self.last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

