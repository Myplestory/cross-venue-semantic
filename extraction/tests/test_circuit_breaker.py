"""Unit tests for CircuitBreaker."""

import pytest
import asyncio
from datetime import datetime, UTC, timedelta
from extraction.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_initial_state():
    """Test circuit breaker starts in CLOSED state."""
    breaker = CircuitBreaker()
    
    assert breaker.get_state() == CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.success_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_successful_call():
    """Test successful call doesn't change state."""
    breaker = CircuitBreaker()
    
    async def success_func():
        return "success"
    
    result = await breaker.call(success_func)
    
    assert result == "success"
    assert breaker.get_state() == CircuitState.CLOSED
    assert breaker.failure_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_failure_opens_circuit():
    """Test circuit opens after failure threshold."""
    config = CircuitBreakerConfig(failure_threshold=3)
    breaker = CircuitBreaker(config)
    
    async def fail_func():
        raise RuntimeError("Test error")
    
    for i in range(3):
        try:
            await breaker.call(fail_func)
        except RuntimeError:
            pass
    
    assert breaker.get_state() == CircuitState.OPEN
    assert breaker.failure_count >= 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_open_rejects_calls():
    """Test open circuit rejects calls immediately."""
    config = CircuitBreakerConfig(failure_threshold=1)
    breaker = CircuitBreaker(config)
    
    async def fail_func():
        raise RuntimeError("Test error")
    
    try:
        await breaker.call(fail_func)
    except RuntimeError:
        pass
    
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(fail_func)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_recovery_to_half_open():
    """Test circuit transitions to HALF_OPEN after recovery timeout."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        recovery_timeout=0.1
    )
    breaker = CircuitBreaker(config)
    
    async def fail_func():
        raise RuntimeError("Test error")
    
    try:
        await breaker.call(fail_func)
    except RuntimeError:
        pass
    
    assert breaker.get_state() == CircuitState.OPEN
    
    await asyncio.sleep(0.15)
    
    async def success_func():
        return "success"
    
    result = await breaker.call(success_func)
    
    assert breaker.get_state() == CircuitState.HALF_OPEN
    assert result == "success"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_half_open_to_closed():
    """Test circuit closes after success threshold in HALF_OPEN."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        recovery_timeout=0.1,
        success_threshold=2
    )
    breaker = CircuitBreaker(config)
    
    async def fail_func():
        raise RuntimeError("Test error")
    
    try:
        await breaker.call(fail_func)
    except RuntimeError:
        pass
    
    await asyncio.sleep(0.15)
    
    async def success_func():
        return "success"
    
    for _ in range(2):
        await breaker.call(success_func)
    
    assert breaker.get_state() == CircuitState.CLOSED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_timeout():
    """Test circuit breaker handles timeouts."""
    config = CircuitBreakerConfig(timeout=0.1)
    breaker = CircuitBreaker(config)
    
    async def slow_func():
        await asyncio.sleep(0.2)
        return "success"
    
    with pytest.raises(RuntimeError, match="timeout"):
        await breaker.call(slow_func)

