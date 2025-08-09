"""
Test basic retry and circuit breaker functionality for Task 4.
"""

import time

import pytest

from backend.common.retry import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    RetryableError,
    RetryConfig,
    retry_with_backoff,
)


def test_circuit_breaker_basic_functionality():
    """Test circuit breaker opens after threshold failures."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

    call_count = 0

    @breaker
    def failing_function():
        nonlocal call_count
        call_count += 1
        raise Exception("Simulated failure")

    # First 3 calls should trigger failures but not open circuit
    for i in range(3):
        with pytest.raises(Exception, match="Simulated failure"):
            failing_function()

    # Circuit should now be open
    assert breaker.state == CircuitState.OPEN

    # Next call should raise CircuitBreakerOpenError without calling function
    with pytest.raises(CircuitBreakerOpenError):
        failing_function()

    # Verify function wasn't called for the circuit breaker error
    assert call_count == 3


def test_circuit_breaker_recovery():
    """Test circuit breaker transitions to half-open and can recover."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)

    @breaker
    def intermittent_function(should_fail=True):
        if should_fail:
            raise Exception("Failure")
        return "Success"

    # Trigger failures to open circuit
    for _ in range(2):
        with pytest.raises(Exception):
            intermittent_function(should_fail=True)

    assert breaker.state == CircuitState.OPEN

    # Wait for timeout to expire
    time.sleep(0.15)

    # Next call should transition to half-open and succeed
    result = intermittent_function(should_fail=False)
    assert result == "Success"
    assert breaker.state == CircuitState.CLOSED


def test_retry_with_backoff_success_after_failures():
    """Test retry decorator succeeds after initial failures."""
    config = RetryConfig(max_attempts=3, base_delay=0.01)

    call_count = 0

    @retry_with_backoff(config=config, exceptions=(RetryableError,))
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RetryableError("Temporary failure")
        return "Success"

    result = flaky_function()
    assert result == "Success"
    assert call_count == 3


def test_retry_with_backoff_exhausts_attempts():
    """Test retry decorator fails after exhausting max attempts."""
    config = RetryConfig(max_attempts=2, base_delay=0.01)

    call_count = 0

    @retry_with_backoff(config=config, exceptions=(RetryableError,))
    def always_failing_function():
        nonlocal call_count
        call_count += 1
        raise RetryableError("Persistent failure")

    with pytest.raises(RetryableError, match="Persistent failure"):
        always_failing_function()

    assert call_count == 2


def test_retry_and_circuit_breaker_integration():
    """Test retry with circuit breaker integration."""
    # Create a circuit breaker with low threshold
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
    config = RetryConfig(max_attempts=5, base_delay=0.01)

    call_count = 0

    # Use RetryableError instead of Exception for the retry decorator
    @retry_with_backoff(config=config, exceptions=(RetryableError,))
    @breaker
    def protected_function():
        nonlocal call_count
        call_count += 1
        raise RetryableError("Service failure")

    # This should trigger the circuit breaker and then stop retrying
    # because CircuitBreakerOpenError is not in the retry exceptions
    with pytest.raises(CircuitBreakerOpenError):
        protected_function()

    # Circuit should be open after 2 failures
    assert breaker.state == CircuitState.OPEN
    # Function should have been called exactly 2 times (the failure threshold)
    assert call_count == 2
