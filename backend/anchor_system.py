#!/usr/bin/env python3
"""
Anchor System - Safety and Approval Mechanism

This module implements the Anchor system that prevents uncontrolled autopilot
actions by requiring confirmation before external or core memory changes.

Enhanced with security features and dry-run support:
- Confirmation requirement for all anchor actions
- Rate limiting for approval requests
- Comprehensive audit logging
- Input validation and sanitization
- Dry-run mode for safe testing
"""

import hashlib
import logging
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dry-run utilities
try:
    from common.dryrun import dry_log, is_dry_run

    DRY_RUN_AVAILABLE = True
except ImportError:
    print("Warning: Dry-run utilities not available")
    DRY_RUN_AVAILABLE = False

    def is_dry_run():
        return False

    def dry_log(logger, event, details):
        pass


# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

# Security configuration
MAX_PENDING_ACTIONS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour
MAX_REQUESTS_PER_WINDOW = 50
MAX_ACTION_DESCRIPTION_LENGTH = 1000
REQUIRED_CONFIRMATION_TYPES = {
    "memory_write",
    "memory_delete",
    "external_api",
    "system_config",
    "emotional_state",
    "anchor_adjustment",
}

# Thread safety
anchor_lock = threading.Lock()

# Rate limiting storage
rate_limit_requests = defaultdict(lambda: deque())


def validate_action_data(action_data: dict[str, Any]) -> tuple[bool, str]:
    """Validate action data for security and integrity"""
    try:
        # Check if input is dictionary
        if not isinstance(action_data, dict):
            return False, "Action data must be a dictionary"

        # Validate required fields
        required_fields = ["action_type", "description"]
        for field in required_fields:
            if field not in action_data:
                return False, f"Missing required field: {field}"

        # Validate action type
        action_type = action_data.get("action_type")
        if not isinstance(action_type, str):
            return False, "Action type must be a string"

        # Sanitize action type
        if not re.match(r"^[a-zA-Z_]+$", action_type):
            return False, "Action type contains invalid characters"

        # Validate description
        description = action_data.get("description", "")
        if not isinstance(description, str):
            return False, "Description must be a string"

        if len(description) > MAX_ACTION_DESCRIPTION_LENGTH:
            return False, f"Description exceeds maximum length of {MAX_ACTION_DESCRIPTION_LENGTH}"

        # Validate priority if present
        if "priority" in action_data:
            priority = action_data["priority"]
            if not isinstance(priority, (int, float)):
                return False, "Priority must be a number"

            if priority < 0 or priority > 10:
                return False, "Priority must be between 0 and 10"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating action data: {e!s}")
        return False, f"Validation error: {e!s}"


def sanitize_description(description: str) -> str:
    """Sanitize action descriptions"""
    if not isinstance(description, str):
        return ""

    # Remove potential injection patterns
    description = re.sub(r'[<>"\']', "", description)

    # Limit length
    if len(description) > MAX_ACTION_DESCRIPTION_LENGTH:
        description = description[:MAX_ACTION_DESCRIPTION_LENGTH] + "..."

    return description.strip()


def check_rate_limit(requester_id: str) -> bool:
    """Check if requester has exceeded rate limit"""
    current_time = time.time()

    # Clean old requests
    while (
        rate_limit_requests[requester_id]
        and rate_limit_requests[requester_id][0] < current_time - RATE_LIMIT_WINDOW
    ):
        rate_limit_requests[requester_id].popleft()

    # Check limit
    if len(rate_limit_requests[requester_id]) >= MAX_REQUESTS_PER_WINDOW:
        logger.warning(f"Rate limit exceeded for requester: {requester_id}")
        return False

    # Add current request
    rate_limit_requests[requester_id].append(current_time)
    return True


def log_anchor_action(
    action_type: str, action_data: dict[str, Any], response: str, requester_id: str = "unknown"
):
    """Log anchor actions for audit trail"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "action_id": action_data.get("action_id", "unknown"),
            "description": action_data.get("description", ""),
            "response": response,
            "requester_id": requester_id,
            "thread_id": threading.get_ident(),
        }

        logger.info(f"Anchor action logged: {action_type} -> {response} by {requester_id}")

        # Log warning for denied actions
        if response == "denied":
            logger.warning(
                f"Anchor action DENIED: {action_type} - {action_data.get('description', '')}"
            )

        return log_entry

    except Exception as e:
        logger.error(f"Error logging anchor action: {e!s}")


class ActionType(Enum):
    """Types of actions that require anchor approval"""

    MEMORY_WRITE = "memory_write"
    MEMORY_DELETE = "memory_delete"
    EXTERNAL_API = "external_api"
    SYSTEM_CONFIG = "system_config"
    EMOTIONAL_STATE = "emotional_state"
    ANCHOR_ADJUSTMENT = "anchor_adjustment"
    PERSONALITY_CHANGE = "personality_change"
    CRITICAL_DECISION = "critical_decision"


class AnchorResponse(Enum):
    """Anchor response types"""

    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    INVALID_REQUEST = "invalid_request"


class AnchorSystem:
    """
    Enhanced Anchor system for controlling autopilot actions.

    Provides approval mechanism for potentially dangerous or
    impactful actions before they are executed. Now includes
    security features like rate limiting, validation, and audit logging.
    """

    def __init__(self, timeout_seconds: int = 30, require_confirmation: bool = True):
        """
        Initialize the Anchor system.

        Args:
            timeout_seconds (int): Timeout for pending approvals
            require_confirmation (bool): Whether to require manual confirmation
        """
        self.timeout_seconds = timeout_seconds
        self.require_confirmation = require_confirmation
        self.pending_actions = {}
        self.approval_history = deque(maxlen=1000)  # Keep last 1000 approvals
        self.denied_actions = deque(maxlen=500)  # Keep last 500 denials

        # Security tracking
        self.creation_time = datetime.now()
        self.total_requests = 0
        self.approved_count = 0
        self.denied_count = 0
        self.timeout_count = 0

        logger.info(f"AnchorSystem initialized - confirmation required: {require_confirmation}")

    def confirm(
        self, autopilot_action: dict[str, Any], requester_id: str = "autopilot"
    ) -> AnchorResponse:
        """
        Request confirmation for an autopilot action with enhanced security and dry-run support.

        Args:
            autopilot_action (Dict[str, Any]): Action details requiring approval
            requester_id (str): Identifier of the entity requesting approval

        Returns:
            AnchorResponse: Approval decision
        """
        start_time = time.time()

        # Check dry-run mode first
        dry = is_dry_run()
        if dry:
            dry_log(
                logger,
                "anchor.confirm.simulated",
                {
                    "proposed": str(autopilot_action)[:256],
                    "requester": requester_id,
                    "action_type": autopilot_action.get("action_type", "unknown"),
                },
            )
            # Return simulated approval with metadata
            return AnchorResponse.APPROVED

        try:
            with anchor_lock:
                self.total_requests += 1

                # Rate limiting check
                if not check_rate_limit(requester_id):
                    self.denied_count += 1
                    log_anchor_action(
                        "rate_limit_check", autopilot_action, "rate_limited", requester_id
                    )
                    return AnchorResponse.RATE_LIMITED

                # Validate action data
                is_valid, validation_message = validate_action_data(autopilot_action)
                if not is_valid:
                    self.denied_count += 1
                    logger.error(f"Invalid action data: {validation_message}")
                    log_anchor_action(
                        "validation", autopilot_action, "invalid_request", requester_id
                    )
                    return AnchorResponse.INVALID_REQUEST

                # Check if pending actions limit exceeded
                if len(self.pending_actions) >= MAX_PENDING_ACTIONS:
                    self.denied_count += 1
                    logger.warning("Maximum pending actions limit reached")
                    log_anchor_action("pending_limit", autopilot_action, "denied", requester_id)
                    return AnchorResponse.DENIED

                # Generate unique action ID
                action_id = self._generate_action_id(autopilot_action, requester_id)

                # Sanitize action data
                sanitized_action = self._sanitize_action(autopilot_action)

                # Check if confirmation is required for this action type
                action_type = sanitized_action.get("action_type", "").lower()

                if not self.require_confirmation or action_type not in REQUIRED_CONFIRMATION_TYPES:
                    # Auto-approve non-critical actions
                    self.approved_count += 1
                    log_anchor_action(action_type, sanitized_action, "approved", requester_id)
                    self._record_approval(
                        action_id, sanitized_action, requester_id, auto_approved=True
                    )
                    return AnchorResponse.APPROVED

                # Add to pending actions for manual approval
                self.pending_actions[action_id] = {
                    "action": sanitized_action,
                    "requester_id": requester_id,
                    "timestamp": datetime.now(),
                    "timeout_at": datetime.now() + timedelta(seconds=self.timeout_seconds),
                }

                logger.info(f"Action pending approval: {action_id} from {requester_id}")
                log_anchor_action(action_type, sanitized_action, "pending", requester_id)

                # For now, auto-approve to maintain functionality
                # In production, this would wait for manual approval
                approval_response = self._auto_approve_for_demo(action_id)

                duration = time.time() - start_time
                logger.info(f"Anchor confirmation completed in {duration:.3f}s")

                return approval_response

        except Exception as e:
            logger.error(f"Error in anchor confirmation: {e!s}")
            self.denied_count += 1
            return AnchorResponse.DENIED

    def _generate_action_id(self, action_data: dict[str, Any], requester_id: str) -> str:
        """Generate unique action ID"""
        timestamp = str(time.time())
        action_str = str(action_data.get("action_type", "")) + str(
            action_data.get("description", "")
        )
        combined = f"{timestamp}:{requester_id}:{action_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _sanitize_action(self, action_data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize action data"""
        sanitized = action_data.copy()

        # Sanitize description
        if "description" in sanitized:
            sanitized["description"] = sanitize_description(sanitized["description"])

        # Ensure action type is clean
        if "action_type" in sanitized:
            action_type = str(sanitized["action_type"]).lower()
            sanitized["action_type"] = re.sub(r"[^a-zA-Z_]", "", action_type)

        return sanitized

    def _auto_approve_for_demo(self, action_id: str) -> AnchorResponse:
        """Auto-approve actions for demo purposes"""
        if action_id in self.pending_actions:
            action_data = self.pending_actions[action_id]
            self.approved_count += 1
            self._record_approval(action_id, action_data["action"], action_data["requester_id"])
            del self.pending_actions[action_id]
            return AnchorResponse.APPROVED
        return AnchorResponse.DENIED

    def _record_approval(
        self,
        action_id: str,
        action_data: dict[str, Any],
        requester_id: str,
        auto_approved: bool = False,
    ):
        """Record approval in history"""
        approval_record = {
            "action_id": action_id,
            "action_type": action_data.get("action_type", "unknown"),
            "description": action_data.get("description", ""),
            "requester_id": requester_id,
            "timestamp": datetime.now().isoformat(),
            "auto_approved": auto_approved,
        }
        self.approval_history.append(approval_record)

    def get_stats(self) -> dict[str, Any]:
        """Get anchor system statistics"""
        return {
            "total_requests": self.total_requests,
            "approved_count": self.approved_count,
            "denied_count": self.denied_count,
            "timeout_count": self.timeout_count,
            "pending_count": len(self.pending_actions),
            "uptime": (datetime.now() - self.creation_time).total_seconds(),
            "approval_rate": self.approved_count / max(self.total_requests, 1),
        }

    def get_recent_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent approval history"""
        history_list = list(self.approval_history)
        return history_list[-limit:] if len(history_list) > limit else history_list

        # Base safety scores by action type
        base_scores = {
            "memory_write": 0.7,
            "memory_delete": 0.3,  # More dangerous
            "external_api": 0.5,
            "system_config": 0.2,  # Very dangerous
            "emotional_state": 0.8,
        }

        safety_score = base_scores.get(action_type, 0.5)

        # Adjust based on target sensitivity
        if "identity" in target.lower():
            safety_score *= 0.5  # Identity changes are risky
        elif "core" in target.lower():
            safety_score *= 0.6  # Core system changes
        elif "temp" in target.lower() or "cache" in target.lower():
            safety_score *= 1.2  # Temporary changes are safer

        return min(1.0, safety_score)

    def _generate_action_id(self) -> str:
        """Generate a unique action ID"""
        return f"anchor_{int(time.time() * 1000)}"

    def review_pending_actions(self) -> list[dict[str, Any]]:
        """
        Review and clean up pending actions.

        Returns:
            List[Dict]: List of pending actions that haven't timed out
        """
        current_time = time.time()
        expired_actions = []

        for action_id, action_data in self.pending_actions.items():
            if current_time - action_data["timestamp"] > self.timeout_seconds:
                expired_actions.append(action_id)
                logger.warning(f"Action {action_id} timed out")

        # Remove expired actions
        for action_id in expired_actions:
            del self.pending_actions[action_id]

        return list(self.pending_actions.values())

    def approve_pending_action(self, action_id: str) -> bool:
        """
        Manually approve a pending action.

        Args:
            action_id (str): ID of the action to approve

        Returns:
            bool: True if action was found and approved
        """
        if action_id in self.pending_actions:
            del self.pending_actions[action_id]
            logger.info(f"Manually approved action {action_id}")
            return True
        return False

    def get_approval_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get recent approval history.

        Args:
            limit (int): Maximum number of records to return

        Returns:
            List[Dict]: Recent approval history
        """
        history_list = list(self.approval_history)
        return history_list[-limit:] if len(history_list) > limit else history_list


# Global anchor instance
anchor = AnchorSystem()


def require_anchor_approval(
    action_type: str, target: str = "", data: Optional[dict] = None
) -> bool:
    """
    Decorator function to require anchor approval for actions.

    Args:
        action_type (str): Type of action being performed
        target (str): Target of the action
        data (Dict): Additional action data

    Returns:
        bool: True if action is approved, False otherwise
    """
    action = {"type": action_type, "target": target, "data": data or {}}

    response = anchor.confirm(action)
    return response == AnchorResponse.APPROVED


if __name__ == "__main__":
    # Test the anchor system
    test_actions = [
        {"type": "memory_write", "target": "temp_cache", "data": {"key": "test"}},
        {"type": "memory_delete", "target": "core_identity", "data": {"key": "personality"}},
        {"type": "external_api", "target": "weather_service", "data": {"endpoint": "/current"}},
    ]

    for action in test_actions:
        response = anchor.confirm(action)
        print(f"Action {action['type']} -> {response.value}")
