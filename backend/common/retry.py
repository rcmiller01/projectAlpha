"""
Retry utilities with backoff and circuit breaker for ProjectAlpha.

Provides decorators for handling failures in external dependencies with
graceful degradation and logging.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable as _Callable
from enum import Enum
from typing import Any, Literal, Optional, cast

# Generic callable taking any args and returning any
AnyCallable = _Callable[..., Any]

logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and calls are blocked."""

    pass


class CircuitState(Enum):
    """States for circuit breaker pattern."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing - circuit is open
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count: int = 0
        self.last_failure_time: float | None = None
        self.state: CircuitState = CircuitState.CLOSED

    def __call__(self, func: AnyCallable) -> AnyCallable:
        """Decorator to apply circuit breaker to a function."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if self._should_attempt_call():
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except self.expected_exception:
                    self._on_failure()
                    raise
            raise CircuitBreakerOpenError(f"Circuit breaker is OPEN. Disabled for {self.timeout}s")

        return cast(AnyCallable, wrapper)

    def _should_attempt_call(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self._timeout_expired():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker -> HALF_OPEN")
                return True
            return False
        # HALF_OPEN
        return True

    def _timeout_expired(self) -> bool:
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout

    def _on_success(self) -> None:
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker -> CLOSED")

    def _on_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                ("Circuit breaker OPEN after %d failures. " "Will retry after %.1fs"),
                self.failure_count,
                self.timeout,
            )

    def get_state(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "timeout": self.timeout,
        }


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
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_range = jitter_range


class RetryableError(Exception):
    pass


class NetworkError(RetryableError):
    pass


class ServiceUnavailableError(RetryableError):
    pass


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Compute delay for attempt with exponential backoff and jitter."""
    delay = config.base_delay * (config.exponential_base**attempt)
    delay = min(delay, config.max_delay)
    if config.jitter:
        jitter_amount = delay * config.jitter_range
        jitter = random.uniform(-jitter_amount, jitter_amount)
        delay = max(0.0, delay + jitter)
    return delay


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple[type[Exception], ...] = (RetryableError,),
    on_retry: Optional[_Callable[[int, Exception], None]] = None,
    on_final_failure: Optional[_Callable[[Exception], Any]] = None,
) -> AnyCallable:
    """Retry functions with exponential backoff."""

    if config is None:
        config = RetryConfig()

    def decorator(func: AnyCallable) -> AnyCallable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception: Exception | None = None
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            "Function %s failed after %d attempts. Error: %s",
                            func.__name__,
                            config.max_attempts,
                            e,
                        )
                        if on_final_failure:
                            return on_final_failure(e)
                        raise
                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        ("Function %s failed on attempt %d/%d. Error: %s. " "Retrying in %.2fs..."),
                        func.__name__,
                        attempt + 1,
                        config.max_attempts,
                        e,
                        delay,
                    )
                    if on_retry:
                        on_retry(attempt + 1, e)
                    time.sleep(delay)
                except Exception as e:
                    logger.error(
                        "Function %s failed with non-retryable error: %s",
                        func.__name__,
                        e,
                    )
                    raise
            if last_exception:
                raise last_exception

        return cast(AnyCallable, wrapper)

    return cast(AnyCallable, decorator)


# Common retry configs
HRM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=True,
)

ARBITER_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=1.5,
    jitter=True,
)

MEMORY_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=0.2,
    max_delay=5.0,
    exponential_base=2.0,
    jitter=True,
)


def retry_hrm_call(func: AnyCallable) -> AnyCallable:
    wrapped = retry_with_backoff(
        config=HRM_RETRY_CONFIG,
        exceptions=(
            NetworkError,
            ServiceUnavailableError,
            ConnectionError,
            TimeoutError,
        ),
        on_retry=lambda a, e: logger.info("HRM call retry #%d: %s", a, e),
        on_final_failure=lambda e: {
            "error": "HRM service unavailable",
            "details": str(e),
        },
    )(func)
    return cast(AnyCallable, wrapped)


def retry_arbiter_call(func: AnyCallable) -> AnyCallable:
    wrapped = retry_with_backoff(
        config=ARBITER_RETRY_CONFIG,
        exceptions=(
            NetworkError,
            ServiceUnavailableError,
            ConnectionError,
            TimeoutError,
        ),
        on_retry=lambda a, e: logger.info("Arbiter call retry #%d: %s", a, e),
        on_final_failure=lambda e: {
            "error": "Arbiter service unavailable",
            "details": str(e),
        },
    )(func)
    return cast(AnyCallable, wrapped)


def retry_memory_call(func: AnyCallable) -> AnyCallable:
    wrapped = retry_with_backoff(
        config=MEMORY_RETRY_CONFIG,
        exceptions=(NetworkError, ServiceUnavailableError, OSError, IOError),
        on_retry=lambda a, e: logger.info("Memory operation retry #%d: %s", a, e),
        on_final_failure=lambda e: {
            "error": "Memory operation failed",
            "details": str(e),
        },
    )(func)
    return cast(AnyCallable, wrapped)


def check_service_health(service_url: str, timeout: float = 5.0) -> bool:
    try:
        import requests

        response = requests.get(f"{service_url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.warning("Health check failed for %s: %s", service_url, e)
        return False


def backoff_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    delay: float = base_delay * (2**attempt)
    return delay if delay <= max_delay else max_delay


class RetryContext:
    """Context manager for retryable operations with logging."""

    def __init__(
        self,
        operation_name: str,
        config: Optional[RetryConfig] = None,
    ) -> None:
        self.operation_name = operation_name
        self.config = config or RetryConfig()
        self.start_time: float | None = None
        self.attempt = 0

    def __enter__(self) -> RetryContext:
        self.start_time = time.time()
        self.attempt += 1
        logger.debug("Starting %s (attempt %d)", self.operation_name, self.attempt)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        duration = time.time() - (self.start_time or 0.0)
        if exc_type is None:
            logger.debug("Completed %s in %.2fs", self.operation_name, duration)
        else:
            logger.warning(
                "Failed %s after %.2fs: %s",
                self.operation_name,
                duration,
                exc_val,
            )
        return False


def retry_with_circuit_breaker(
    retry_config: Optional[RetryConfig] = None,
    circuit_config: Optional[dict[str, Any]] = None,
    exceptions: tuple[type[Exception], ...] = (RetryableError,),
    on_retry: Optional[_Callable[[int, Exception], None]] = None,
    on_final_failure: Optional[_Callable[[Exception], Any]] = None,
) -> AnyCallable:
    """Combine retry with a per-function circuit breaker."""

    if retry_config is None:
        retry_config = RetryConfig()
    if circuit_config is None:
        circuit_config = {"failure_threshold": 5, "timeout": 60.0}

    def decorator(func: AnyCallable) -> AnyCallable:
        circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_config.get("failure_threshold", 5),
            timeout=circuit_config.get("timeout", 60.0),
            expected_exception=exceptions[0] if exceptions else Exception,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not circuit_breaker._should_attempt_call():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN for {func.__name__}. "
                    f"Disabled for {circuit_breaker.timeout}s"
                )

            for attempt in range(retry_config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    circuit_breaker._on_success()
                    return result
                except exceptions as e:
                    circuit_breaker._on_failure()
                    if attempt == retry_config.max_attempts - 1:
                        logger.error(
                            "Function %s failed after %d attempts. Error: %s",
                            func.__name__,
                            retry_config.max_attempts,
                            e,
                        )
                        if on_final_failure:
                            return on_final_failure(e)
                        raise
                    delay = calculate_delay(attempt, retry_config)
                    logger.warning(
                        ("Function %s failed on attempt %d/%d. Error: %s. " "Retrying in %.2fs..."),
                        func.__name__,
                        attempt + 1,
                        retry_config.max_attempts,
                        e,
                        delay,
                    )
                    if on_retry:
                        on_retry(attempt + 1, e)
                    time.sleep(delay)
                except Exception as e:
                    circuit_breaker._on_failure()
                    logger.error(
                        "Function %s failed with non-retryable error: %s",
                        func.__name__,
                        e,
                    )
                    raise

        # Attach for monitoring; use setattr to keep type-checkers happy
        wrapper._circuit_breaker = circuit_breaker  # type: ignore[attr-defined]
        return cast(AnyCallable, wrapper)

    return cast(AnyCallable, decorator)
