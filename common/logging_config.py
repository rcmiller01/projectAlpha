"""
Centralized logging configuration for ProjectAlpha.
Provides standardized logging format: [TIMESTAMP] [MODULE] ACTION - DETAILS
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class ProjectAlphaFormatter(logging.Formatter):
    """Custom formatter for ProjectAlpha standardized log format."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with standardized format."""
        # Get timestamp in ISO format
        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract module name from record
        module_name = record.name.upper().replace(".", "_")
        if module_name.startswith("BACKEND_"):
            module_name = module_name[8:]  # Remove "BACKEND_" prefix

        # Determine action from record
        action = getattr(record, "action", record.levelname)

        # Format message
        message = record.getMessage()

        # Combine into standard format
        formatted = f"[{timestamp}] [{module_name}] {action} - {message}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_project_logging(
    log_level: str = "INFO", log_file: Optional[str] = None, enable_console: bool = True
) -> None:
    """
    Setup standardized logging configuration for ProjectAlpha.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_console: Whether to enable console logging
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = ProjectAlphaFormatter()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)

    # Setup module-specific loggers with consistent naming
    module_loggers = [
        "hrm_api",
        "hrm_router",
        "anchor_system",
        "policy_dsl",
        "moe_arbitration",
        "slim_contracts",
        "persona_router",
        "affect_aware_routing",
    ]

    for module_name in module_loggers:
        logger = logging.getLogger(module_name)
        logger.setLevel(getattr(logging, log_level.upper()))
        # Propagate to root logger for consistent formatting
        logger.propagate = True


def log_action(
    logger: logging.Logger,
    action: str,
    details: Optional[str] = None,
    level: str = "INFO",
    **kwargs: Any,
) -> None:
    """
    Log an action with standardized format.

    Args:
        logger: Logger instance to use
        action: Action being performed
        details: Additional details about the action
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        **kwargs: Additional context to include in log
    """
    # Build message
    message_parts = []
    if details:
        message_parts.append(details)

    # Add kwargs as key=value pairs
    if kwargs:
        context_parts = [f"{k}={v}" for k, v in kwargs.items()]
        message_parts.append(" ".join(context_parts))

    message = " ".join(message_parts) if message_parts else ""

    # Create log record with action
    log_level = getattr(logging, level.upper())
    record = logger.makeRecord(
        logger.name,
        log_level,
        "",
        0,
        message,
        (),
        None,  # pathname  # lineno  # args  # exc_info
    )
    record.action = action

    logger.handle(record)


def log_hrm_operation(
    operation: str,
    layer: str,
    success: bool,
    details: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
) -> None:
    """
    Log HRM operation with standardized format.

    Args:
        operation: Operation type (READ, WRITE, DELETE)
        layer: HRM layer (identity, beliefs, ephemeral)
        success: Whether operation succeeded
        details: Additional operation details
        dry_run: Whether this was a dry-run operation
    """
    logger = logging.getLogger("hrm_api")

    status = "SUCCESS" if success else "FAILED"
    if dry_run:
        status = f"DRY_RUN_{status}"

    action = f"{operation}_{layer.upper()}_{status}"

    detail_str = ""
    if details:
        detail_str = " ".join([f"{k}={v}" for k, v in details.items()])

    log_action(logger, action, detail_str)


def log_moe_arbitration(
    experts: list,
    selected_expert: str,
    confidence: float,
    context: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
) -> None:
    """
    Log MoE arbitration decision with standardized format.

    Args:
        experts: List of available experts
        selected_expert: Expert selected by arbitration
        confidence: Confidence score for selection
        context: Additional arbitration context
        dry_run: Whether this was a dry-run arbitration
    """
    logger = logging.getLogger("moe_arbitration")

    action = "DRY_RUN_ARBITRATION" if dry_run else "ARBITRATION"

    details = f"selected={selected_expert} confidence={confidence:.3f} experts={len(experts)}"

    if context:
        context_str = " ".join([f"{k}={v}" for k, v in context.items()])
        details += f" {context_str}"

    log_action(logger, action, details)


def log_anchor_decision(
    operation: str, approved: bool, reason: str, requester: str = "unknown", dry_run: bool = False
) -> None:
    """
    Log anchor system decision with standardized format.

    Args:
        operation: Operation requiring anchor approval
        approved: Whether operation was approved
        reason: Reason for approval/denial
        requester: Entity requesting approval
        dry_run: Whether this was a dry-run check
    """
    logger = logging.getLogger("anchor_system")

    status = "APPROVED" if approved else "DENIED"
    if dry_run:
        status = f"DRY_RUN_{status}"

    action = f"ANCHOR_{status}"

    details = f"operation={operation} requester={requester} reason={reason}"

    log_action(logger, action, details)


# Initialize logging if this module is imported
if __name__ != "__main__":
    # Auto-setup with environment variables
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE")
    enable_console = os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"

    setup_project_logging(log_level, log_file, enable_console)
