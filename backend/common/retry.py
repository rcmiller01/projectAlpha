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
from typing import Any, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


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
    )(func)


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
    )(func)


def retry_memory_call(func: Callable) -> Callable:
    """Decorator for memory-related calls with appropriate retry settings."""
    return retry_with_backoff(
        config=MEMORY_RETRY_CONFIG,
        exceptions=(NetworkError, ServiceUnavailableError, OSError, IOError),
        on_retry=lambda attempt, error: logger.info(f"Memory operation retry #{attempt}: {error}"),
        on_final_failure=lambda error: {"error": "Memory operation failed", "details": str(error)},
    )(func)


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
    delay = base_delay * (2**attempt)
    return min(delay, max_delay)


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
