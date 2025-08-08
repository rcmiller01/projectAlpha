"""
Retry utilities with exponential backoff and jitter for ProjectAlpha.

Provides decorators for handling failures in external dependencies with
graceful degradation and proper logging.
"""

import functools
import logging
import random
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States for circuit breaker pattern."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing - circuit is open
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for handling cascading failures.

    Monitors failure rates and opens the circuit when failure threshold is
    exceeded,
    allowing the system to fail fast and recover gracefully.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures to open circuit
            timeout: Seconds to wait before attempting to close circuit
            expected_exception: Exception type that counts as a failure
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count: int = 0
        self.last_failure_time: float | None = None
        self.state: CircuitState = CircuitState.CLOSED

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self._should_attempt_call():
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except self.expected_exception:
                    self._on_failure()
                    raise
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Calls disabled for {self.timeout}s"
                )

        return wrapper

    def _should_attempt_call(self) -> bool:
        """Check if we should attempt the call based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._timeout_expired():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False

        # HALF_OPEN state
        return True

    def _timeout_expired(self) -> bool:
        """Check if timeout has expired since last failure."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker transitioning to CLOSED")

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN after {self.failure_count} failures. "
                f"Will retry after {self.timeout}s"
            )

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state for monitoring."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "timeout": self.timeout,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and calls are blocked."""

    pass


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_range: float = 0.1,
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter
            jitter_range: Range for jitter (0.0 to 1.0)
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_range = jitter_range


class RetryableError(Exception):
    """Base class for errors that should trigger retries."""

    pass


class NetworkError(RetryableError):
    """Network-related errors that should be retried."""

    pass


class ServiceUnavailableError(RetryableError):
    """Service unavailable errors that should be retried."""

    pass


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for retry attempt with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-based)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff
    delay = config.base_delay * (config.exponential_base**attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay)

    # Add jitter if enabled
    if config.jitter:
        jitter_amount = delay * config.jitter_range
        jitter = random.uniform(-jitter_amount, jitter_amount)
        delay = max(0, delay + jitter)

    return delay


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple[type[Exception], ...] = (RetryableError,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    on_final_failure: Optional[Callable[[Exception], Any]] = None,
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        config: Retry configuration (uses default if None)
        exceptions: Tuple of exception types to retry on
        on_retry: Callback called on each retry (attempt_num, exception)
        on_final_failure: Callback called when all retries are exhausted

    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        # Final attempt failed
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts. "
                            f"Final error: {e!s}"
                        )
                        if on_final_failure:
                            return on_final_failure(e)
                        raise

                    # Calculate delay for next attempt
                    delay = calculate_delay(attempt, config)

                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{config.max_attempts}. "
                        f"Error: {e!s}. Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(attempt + 1, e)

                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Function {func.__name__} failed with non-retryable error: {e!s}")
                    raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


# Predefined configurations for common scenarios
HRM_RETRY_CONFIG = RetryConfig(
    max_attempts=3, base_delay=0.5, max_delay=10.0, exponential_base=2.0, jitter=True
)

ARBITER_RETRY_CONFIG = RetryConfig(
    max_attempts=5, base_delay=1.0, max_delay=30.0, exponential_base=1.5, jitter=True
)

MEMORY_RETRY_CONFIG = RetryConfig(
    max_attempts=2, base_delay=0.2, max_delay=5.0, exponential_base=2.0, jitter=True
)


# Convenience decorators for common scenarios
def retry_hrm_call(func: Callable) -> Callable:
    """Decorator for HRM-related calls with appropriate retry settings."""
    return retry_with_backoff(
        config=HRM_RETRY_CONFIG,
        exceptions=(NetworkError, ServiceUnavailableError, ConnectionError, TimeoutError),
        on_retry=lambda attempt, error: logger.info(f"HRM call retry #{attempt}: {error}"),
        on_final_failure=lambda error: {"error": "HRM service unavailable", "details": str(error)},
    )(func)  # type: ignore[no-any-return]


def retry_arbiter_call(func: Callable) -> Callable:
    """Decorator for CoreArbiter-related calls with appropriate retry settings."""
    return retry_with_backoff(
        config=ARBITER_RETRY_CONFIG,
        exceptions=(NetworkError, ServiceUnavailableError, ConnectionError, TimeoutError),
        on_retry=lambda attempt, error: logger.info(f"Arbiter call retry #{attempt}: {error}"),
        on_final_failure=lambda error: {
            "error": "Arbiter service unavailable",
            "details": str(error),
        },
    )(func)  # type: ignore[no-any-return]


def retry_memory_call(func: Callable) -> Callable:
    """Decorator for memory-related calls with appropriate retry settings."""
    return retry_with_backoff(
        config=MEMORY_RETRY_CONFIG,
        exceptions=(NetworkError, ServiceUnavailableError, OSError, IOError),
        on_retry=lambda attempt, error: logger.info(f"Memory operation retry #{attempt}: {error}"),
        on_final_failure=lambda error: {"error": "Memory operation failed", "details": str(error)},
    )(func)  # type: ignore[no-any-return]


# Utility functions for checking service health
def check_service_health(service_url: str, timeout: float = 5.0) -> bool:
    """
    Check if a service is healthy and responding.

    Args:
        service_url: URL to check
        timeout: Request timeout in seconds

    Returns:
        True if service is healthy, False otherwise
    """
    try:
        import requests

        response = requests.get(f"{service_url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Health check failed for {service_url}: {e}")
        return False


def backoff_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Simple exponential backoff delay calculation.

    Args:
        attempt: Attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds
    """
    delay: float = base_delay * (2**attempt)
    return delay if delay <= max_delay else max_delay


# Context manager for handling retryable operations
class RetryContext:
    """Context manager for retryable operations with logging."""

    def __init__(self, operation_name: str, config: Optional[RetryConfig] = None):
        self.operation_name = operation_name
        self.config = config or RetryConfig()
        self.start_time = None
        self.attempt = 0

    def __enter__(self):
        self.start_time = time.time()
        self.attempt += 1
        logger.debug(f"Starting {self.operation_name} (attempt {self.attempt})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - (self.start_time or 0)

        if exc_type is None:
            logger.debug(f"Completed {self.operation_name} in {duration:.2f}s")
        else:
            logger.warning(f"Failed {self.operation_name} after {duration:.2f}s: {exc_val}")

        return False  # Don't suppress exceptions


def retry_with_circuit_breaker(
    retry_config: Optional[RetryConfig] = None,
    circuit_config: Optional[dict[str, Any]] = None,
    exceptions: tuple[type[Exception], ...] = (RetryableError,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    on_final_failure: Optional[Callable[[Exception], Any]] = None,
):
    """
    Combined decorator for retry with circuit breaker protection.

    Args:
        retry_config: Retry configuration
        circuit_config: Circuit breaker configuration (failure_threshold, timeout, etc.)
        exceptions: Exception types to retry on
        on_retry: Callback on each retry
        on_final_failure: Callback on final failure
    """
    if retry_config is None:
        retry_config = RetryConfig()

    if circuit_config is None:
        circuit_config = {"failure_threshold": 5, "timeout": 60.0}

    def decorator(func: Callable) -> Callable:
        # Create circuit breaker for this function
        circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_config.get("failure_threshold", 5),
            timeout=circuit_config.get("timeout", 60.0),
            expected_exception=exceptions[0] if exceptions else Exception,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check circuit breaker first
            if not circuit_breaker._should_attempt_call():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN for {func.__name__}. "
                    f"Service calls disabled for {circuit_breaker.timeout}s"
                )

            # Apply retry logic
            for attempt in range(retry_config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    circuit_breaker._on_success()
                    return result
                except exceptions as e:
                    circuit_breaker._on_failure()

                    if attempt == retry_config.max_attempts - 1:
                        # Final attempt failed
                        logger.error(
                            f"Function {func.__name__} failed after "
                            f"{retry_config.max_attempts} attempts. Final error: {e!s}"
                        )
                        if on_final_failure:
                            return on_final_failure(e)
                        raise

                    # Calculate delay for next attempt
                    delay = calculate_delay(attempt, retry_config)

                    logger.warning(
                        f"Function {func.__name__} failed on attempt "
                        f"{attempt + 1}/{retry_config.max_attempts}. "
                        f"Error: {e!s}. Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(attempt + 1, e)

                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception
                    circuit_breaker._on_failure()
                    logger.error(
                        f"Function {func.__name__} failed with non-retryable error: {e!s}"
                    )
                    raise

        # Attach circuit breaker for monitoring
        wrapper._circuit_breaker = circuit_breaker  # type: ignore[attr-defined]
        return wrapper

    return decorator
