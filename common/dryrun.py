# common/dryrun.py
"""
Dry-run utility for safe testing and simulation across ProjectAlpha.
Provides consistent dry-run behavior for HRM, MoE, SLiM, and Anchor systems.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict


def _env_truthy(v: str | None) -> bool:
    """Check if environment variable represents a truthy value."""
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def is_dry_run() -> bool:
    """
    Check if global dry-run mode is enabled via environment.
    Global kill-switch via env; routers can still override per-request.
    """
    return _env_truthy(os.getenv("DRY_RUN"))


def as_bool(val: Any) -> bool:
    """Convert various types to boolean for dry-run flag evaluation."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int | float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return False


def merge_dry_run(global_env: bool, request_flag: Any) -> bool:
    """
    Merge global environment dry-run with per-request flag.
    request_flag can be query/header/body param.
    """
    return global_env or as_bool(request_flag)


def dry_log(logger: logging.Logger, event: str, details: dict[str, Any]) -> None:
    """Log dry-run events with consistent structure."""
    logger.info({"event": event, "dry_run": True, **details})


@contextmanager
def dry_guard(logger: logging.Logger, event: str, details: dict[str, Any]):
    """
    Context manager for blocks where you would normally perform side-effects.
    If dry-run, just log and skip the effect.

    Usage:
        with dry_guard(logger, "hrm.write", {"layer": "beliefs"}) as dry:
            if not dry:
                # perform actual write operation
                pass

    Args:
        logger: Logger instance for dry-run logging
        event: Event name for logging
        details: Additional details to include in log

    Yields:
        bool: True if in dry-run mode (skip operations), False otherwise
    """
    dry = is_dry_run()
    if dry:
        dry_log(logger, f"{event}.skipped", details)
        yield True  # tell caller we are in dry mode
    else:
        yield False


def get_dry_run_status() -> dict[str, Any]:
    """Get current dry-run status for debugging/monitoring."""
    return {
        "dry_run_enabled": is_dry_run(),
        "env_variable": os.getenv("DRY_RUN"),
        "source": "environment",
    }


def format_dry_run_response(
    data: dict[str, Any], dry_run: bool, status_code: int | None = None
) -> dict[str, Any]:
    """Format API response with dry-run metadata."""
    response: dict[str, Any] = {
        **data,
        "dry_run": dry_run,
        "status": data.get("status", "simulated" if dry_run else "executed"),
    }

    if dry_run and status_code is None:
        # Default to 202 Accepted for dry-run responses
        response["_suggested_status_code"] = 202
    elif status_code:
        response["_suggested_status_code"] = status_code

    return response


class DryRunConfig:
    """Configuration class for dry-run behavior customization."""

    def __init__(self):
        self.global_enabled = is_dry_run()
        self.log_all_decisions = _env_truthy(os.getenv("DRY_RUN_LOG_ALL", "true"))
        self.simulate_delays = _env_truthy(os.getenv("DRY_RUN_SIMULATE_DELAYS", "false"))
        self.fail_rate = float(os.getenv("DRY_RUN_FAIL_RATE", "0.0"))

    def should_simulate_failure(self) -> bool:
        """Randomly simulate failures for testing (if configured)."""
        if self.fail_rate <= 0:
            return False
        import random

        return random.random() < self.fail_rate

    def get_simulated_delay(self) -> float:
        """Get simulated delay for operations (if configured)."""
        if not self.simulate_delays:
            return 0.0
        # Simulate realistic delays for different operation types
        import random

        return random.uniform(0.1, 0.5)  # 100-500ms


# Global config instance
dry_run_config = DryRunConfig()
