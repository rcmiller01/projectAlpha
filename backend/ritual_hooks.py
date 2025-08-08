"""
Ritual Hooks System for AI Agent Lifecycle Events

Enhanced with security features:
- Ritual hook security with event validation
- Hook registration and execution monitoring
- Session management and access control for ritual operations
- Rate limiting and monitoring for hook executions
"""

import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


class RitualEvent(Enum):
    """Defined ritual events in the AI agent lifecycle"""

    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TASK_BEGIN = "task_begin"
    TASK_COMPLETE = "task_complete"
    ERROR_OCCURRED = "error_occurred"
    MEMORY_UPDATE = "memory_update"
    EMOTION_CHANGE = "emotion_change"
    DECISION_MADE = "decision_made"


class HookPriority(Enum):
    """Hook execution priority levels"""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


# Security configuration
RITUAL_SESSION_LENGTH = 32
MAX_HOOK_NAME_LENGTH = 100
MAX_HOOKS_PER_EVENT = 50
MAX_HOOK_EXECUTION_TIME = 30.0  # seconds
RITUAL_RATE_LIMIT = 100  # hook executions per hour per session
MAX_EVENT_DATA_SIZE = 10000

# Thread safety
ritual_lock = threading.Lock()

# Session management
ritual_sessions = {}
session_expiry_hours = 24

# Rate limiting
ritual_requests = defaultdict(lambda: deque())

# Access monitoring
ritual_access_history = deque(maxlen=1000)


@dataclass
class RitualHook:
    """A ritual hook with metadata and security information"""

    name: str
    event: RitualEvent
    callback: Callable[[dict[str, Any]], Any]
    priority: HookPriority
    created_at: str
    execution_count: int = 0
    last_execution: Optional[str] = None
    session_token: str = ""
    enabled: bool = True


def generate_ritual_session() -> str:
    """Generate a secure ritual session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"ritual:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:RITUAL_SESSION_LENGTH]


def validate_ritual_session(session_token: str) -> bool:
    """Validate ritual session token"""
    if not session_token or len(session_token) != RITUAL_SESSION_LENGTH:
        return False

    if session_token not in ritual_sessions:
        return False

    # Check if session has expired
    session_data = ritual_sessions[session_token]
    if datetime.now() > session_data["expires_at"]:
        del ritual_sessions[session_token]
        return False

    # Update last access time
    session_data["last_access"] = datetime.now()
    return True


def check_ritual_rate_limit(session_token: str) -> bool:
    """Check if ritual operation rate limit is exceeded"""
    current_time = time.time()

    # Clean old requests
    while (
        ritual_requests[session_token] and ritual_requests[session_token][0] < current_time - 3600
    ):  # 1 hour window
        ritual_requests[session_token].popleft()

    # Check limit
    if len(ritual_requests[session_token]) >= RITUAL_RATE_LIMIT:
        logger.warning(f"Ritual rate limit exceeded for session: {session_token[:8]}...")
        return False

    # Add current request
    ritual_requests[session_token].append(current_time)
    return True


def validate_hook_data(name: str, event: RitualEvent, callback: Callable) -> tuple[bool, str]:
    """Validate ritual hook data"""
    try:
        # Validate name
        if not isinstance(name, str):
            return False, "Hook name must be a string"

        if len(name) > MAX_HOOK_NAME_LENGTH:
            return False, f"Hook name too long (max {MAX_HOOK_NAME_LENGTH} characters)"

        # Check for dangerous characters in name
        if re.search(r'[<>"\']', name):
            return False, "Hook name contains invalid characters"

        # Validate event
        if not isinstance(event, RitualEvent):
            return False, "Event must be a valid RitualEvent"

        # Validate callback
        if not callable(callback):
            return False, "Callback must be callable"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating hook data: {e!s}")
        return False, f"Validation error: {e!s}"


def validate_event_data(event_data: dict[str, Any]) -> tuple[bool, str]:
    """Validate event data"""
    try:
        # Check data size
        data_size = len(json.dumps(event_data, default=str))
        if data_size > MAX_EVENT_DATA_SIZE:
            return False, f"Event data too large: {data_size} bytes (max {MAX_EVENT_DATA_SIZE})"

        # Check for dangerous content
        data_str = json.dumps(event_data, default=str)
        dangerous_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"javascript:",  # JavaScript protocol
            r"on\w+\s*=",  # Event handlers
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                return False, "Event data contains potentially dangerous content"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating event data: {e!s}")
        return False, f"Validation error: {e!s}"


def sanitize_hook_name(name: str) -> str:
    """Sanitize hook name for safety"""
    # Remove dangerous characters
    clean_name = re.sub(r'[<>"\']', "", str(name))
    # Limit length
    if len(clean_name) > MAX_HOOK_NAME_LENGTH:
        clean_name = clean_name[:MAX_HOOK_NAME_LENGTH]
    return clean_name


def log_ritual_activity(
    activity_type: str, session_token: str, details: dict[str, Any], status: str = "success"
):
    """Log ritual access activities"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "activity_type": activity_type,
            "session": session_token[:8] + "..." if session_token else "none",
            "details": details,
            "status": status,
            "thread_id": threading.get_ident(),
        }

        ritual_access_history.append(log_entry)

        logger.info(f"Ritual access logged: {activity_type} ({status})")

        if status != "success":
            logger.warning(f"Ritual access issue: {activity_type} failed with {status}")

    except Exception as e:
        logger.error(f"Error logging ritual access: {e!s}")


class RitualHookManager:
    """
    Secure ritual hook manager for AI agent lifecycle events.

    Features:
    - Event-driven hook system with validation
    - Hook registration and execution monitoring
    - Session-based authentication and authorization
    - Rate limiting and access monitoring
    - Comprehensive audit logging
    - Thread-safe concurrent operations
    """

    def __init__(self, session_token: Optional[str] = None):
        """Initialize ritual hook manager with security features"""
        self.session_token = session_token or self.create_session()
        self.creation_time = datetime.now()
        self.hooks: dict[RitualEvent, list[RitualHook]] = {}
        self.execution_count = 0

        # Initialize hook storage for each event
        for event in RitualEvent:
            self.hooks[event] = []

        log_ritual_activity(
            "initialization",
            self.session_token,
            {
                "creation_time": self.creation_time.isoformat(),
                "events_initialized": len(RitualEvent),
            },
        )

        logger.info("RitualHookManager initialized with security features")

    def create_session(self) -> str:
        """Create a new ritual session"""
        with ritual_lock:
            session_token = generate_ritual_session()
            ritual_sessions[session_token] = {
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(hours=session_expiry_hours),
                "last_access": datetime.now(),
                "ritual_operations": 0,
            }
            return session_token

    def validate_session(self, session_token: Optional[str] = None) -> bool:
        """Validate session for ritual operations"""
        token_to_validate = session_token or self.session_token

        if not token_to_validate:
            logger.warning("No session token provided for ritual validation")
            return False

        return validate_ritual_session(token_to_validate)

    def register_hook(
        self,
        name: str,
        event: RitualEvent,
        callback: Callable[[dict[str, Any]], Any],
        priority: HookPriority = HookPriority.NORMAL,
        session_token: Optional[str] = None,
    ) -> bool:
        """
        Register a ritual hook with security validation.

        Args:
            name: Unique name for the hook
            event: Event type to hook into
            callback: Function to call when event occurs
            priority: Execution priority
            session_token: Session token for authentication

        Returns:
            Success status
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_ritual_activity(
                    "register_hook",
                    session_token or self.session_token,
                    {"name": name, "event": event.value, "status": "session_invalid"},
                    "failed",
                )
                return False

            current_token = session_token or self.session_token

            # Check rate limit
            if not check_ritual_rate_limit(current_token):
                log_ritual_activity(
                    "register_hook",
                    current_token,
                    {"name": name, "event": event.value, "status": "rate_limited"},
                    "failed",
                )
                return False

            # Validate hook data
            is_valid, validation_message = validate_hook_data(name, event, callback)
            if not is_valid:
                logger.error(f"Invalid hook data: {validation_message}")
                log_ritual_activity(
                    "register_hook",
                    current_token,
                    {"name": name, "error": validation_message},
                    "validation_failed",
                )
                return False

            # Sanitize name
            clean_name = sanitize_hook_name(name)

            with ritual_lock:
                # Check if too many hooks for this event
                if len(self.hooks[event]) >= MAX_HOOKS_PER_EVENT:
                    logger.error(
                        f"Too many hooks for event {event.value} (max {MAX_HOOKS_PER_EVENT})"
                    )
                    log_ritual_activity(
                        "register_hook",
                        current_token,
                        {"name": clean_name, "event": event.value, "error": "too_many_hooks"},
                        "failed",
                    )
                    return False

                # Check for duplicate name
                existing_names = [hook.name for hook in self.hooks[event]]
                if clean_name in existing_names:
                    logger.error(
                        f"Hook with name '{clean_name}' already exists for event {event.value}"
                    )
                    log_ritual_activity(
                        "register_hook",
                        current_token,
                        {"name": clean_name, "event": event.value, "error": "duplicate_name"},
                        "failed",
                    )
                    return False

                # Create and register hook
                hook = RitualHook(
                    name=clean_name,
                    event=event,
                    callback=callback,
                    priority=priority,
                    created_at=datetime.now().isoformat(),
                    session_token=current_token[:8] + "...",
                )

                self.hooks[event].append(hook)

                # Sort hooks by priority
                self.hooks[event].sort(key=lambda h: h.priority.value)

                # Update session tracking
                if current_token in ritual_sessions:
                    ritual_sessions[current_token]["ritual_operations"] += 1

            log_ritual_activity(
                "register_hook",
                current_token,
                {
                    "name": clean_name,
                    "event": event.value,
                    "priority": priority.value,
                    "hooks_count": len(self.hooks[event]),
                },
            )

            logger.info(
                f"Hook registered: {clean_name} for {event.value} (priority: {priority.value})"
            )
            return True

        except Exception as e:
            logger.error(f"Error registering hook: {e!s}")
            log_ritual_activity(
                "register_hook",
                session_token or self.session_token,
                {"name": name, "error": str(e)},
                "error",
            )
            return False

    def trigger_event(
        self,
        event: RitualEvent,
        event_data: Optional[dict[str, Any]] = None,
        session_token: Optional[str] = None,
    ) -> list[Any]:
        """
        Trigger an event and execute all registered hooks with security validation.

        Args:
            event: Event to trigger
            event_data: Data to pass to hooks
            session_token: Session token for authentication

        Returns:
            List of hook execution results
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_ritual_activity(
                    "trigger_event",
                    session_token or self.session_token,
                    {"event": event.value, "status": "session_invalid"},
                    "failed",
                )
                return []

            current_token = session_token or self.session_token

            # Check rate limit
            if not check_ritual_rate_limit(current_token):
                log_ritual_activity(
                    "trigger_event",
                    current_token,
                    {"event": event.value, "status": "rate_limited"},
                    "failed",
                )
                return []

            # Validate event data
            if event_data is None:
                event_data = {}

            is_valid, validation_message = validate_event_data(event_data)
            if not is_valid:
                logger.error(f"Invalid event data: {validation_message}")
                log_ritual_activity(
                    "trigger_event",
                    current_token,
                    {"event": event.value, "error": validation_message},
                    "validation_failed",
                )
                return []

            # Add security metadata to event data
            secure_event_data = event_data.copy()
            secure_event_data["_ritual_meta"] = {
                "event": event.value,
                "timestamp": datetime.now().isoformat(),
                "session_token": current_token[:8] + "...",
                "execution_id": hashlib.md5(f"{current_token}{time.time()}".encode()).hexdigest()[
                    :16
                ],
            }

            results = []
            hooks_executed = 0

            # Execute hooks in priority order
            for hook in self.hooks.get(event, []):
                if not hook.enabled:
                    continue

                try:
                    start_time = time.time()

                    # Execute hook with timeout protection
                    result = hook.callback(secure_event_data)

                    execution_time = time.time() - start_time

                    # Check execution time
                    if execution_time > MAX_HOOK_EXECUTION_TIME:
                        logger.warning(
                            f"Hook {hook.name} exceeded execution time limit: {execution_time:.2f}s"
                        )

                    # Update hook statistics
                    with ritual_lock:
                        hook.execution_count += 1
                        hook.last_execution = datetime.now().isoformat()
                        self.execution_count += 1

                    results.append(
                        {
                            "hook_name": hook.name,
                            "result": result,
                            "execution_time": execution_time,
                            "success": True,
                        }
                    )

                    hooks_executed += 1

                    logger.debug(
                        f"Hook executed: {hook.name} for {event.value} ({execution_time:.3f}s)"
                    )

                except Exception as hook_error:
                    logger.error(f"Error executing hook {hook.name}: {hook_error!s}")
                    results.append(
                        {"hook_name": hook.name, "error": str(hook_error), "success": False}
                    )

            log_ritual_activity(
                "trigger_event",
                current_token,
                {
                    "event": event.value,
                    "hooks_executed": hooks_executed,
                    "total_hooks": len(self.hooks.get(event, [])),
                    "execution_count": self.execution_count,
                },
            )

            logger.info(f"Event triggered: {event.value} ({hooks_executed} hooks executed)")
            return results

        except Exception as e:
            logger.error(f"Error triggering event: {e!s}")
            log_ritual_activity(
                "trigger_event",
                session_token or self.session_token,
                {"event": event.value, "error": str(e)},
                "error",
            )
            return []

    def unregister_hook(
        self, name: str, event: RitualEvent, session_token: Optional[str] = None
    ) -> bool:
        """
        Unregister a ritual hook with security validation.

        Args:
            name: Name of the hook to remove
            event: Event type the hook is registered for
            session_token: Session token for authentication

        Returns:
            Success status
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_ritual_activity(
                    "unregister_hook",
                    session_token or self.session_token,
                    {"name": name, "event": event.value, "status": "session_invalid"},
                    "failed",
                )
                return False

            current_token = session_token or self.session_token
            clean_name = sanitize_hook_name(name)

            with ritual_lock:
                hooks_list = self.hooks.get(event, [])
                for i, hook in enumerate(hooks_list):
                    if hook.name == clean_name:
                        removed_hook = hooks_list.pop(i)

                        log_ritual_activity(
                            "unregister_hook",
                            current_token,
                            {
                                "name": clean_name,
                                "event": event.value,
                                "execution_count": removed_hook.execution_count,
                            },
                        )

                        logger.info(f"Hook unregistered: {clean_name} from {event.value}")
                        return True

                log_ritual_activity(
                    "unregister_hook",
                    current_token,
                    {"name": clean_name, "event": event.value, "error": "not_found"},
                    "failed",
                )
                return False

        except Exception as e:
            logger.error(f"Error unregistering hook: {e!s}")
            log_ritual_activity(
                "unregister_hook",
                session_token or self.session_token,
                {"name": name, "error": str(e)},
                "error",
            )
            return False

    def get_hook_stats(self) -> dict[str, Any]:
        """Get ritual hook statistics"""
        hook_counts = {event.value: len(hooks) for event, hooks in self.hooks.items()}
        total_hooks = sum(hook_counts.values())

        return {
            "session_token": self.session_token[:8] + "..." if self.session_token else None,
            "creation_time": self.creation_time.isoformat(),
            "total_hooks": total_hooks,
            "hooks_by_event": hook_counts,
            "total_executions": self.execution_count,
            "events_available": [event.value for event in RitualEvent],
        }

    def list_hooks(self, event: Optional[RitualEvent] = None) -> dict[str, list[dict[str, Any]]]:
        """List registered hooks"""
        result = {}

        events_to_list = [event] if event else list(RitualEvent)

        for evt in events_to_list:
            hooks_info = []
            for hook in self.hooks.get(evt, []):
                hooks_info.append(
                    {
                        "name": hook.name,
                        "priority": hook.priority.value,
                        "execution_count": hook.execution_count,
                        "last_execution": hook.last_execution,
                        "enabled": hook.enabled,
                        "created_at": hook.created_at,
                    }
                )
            result[evt.value] = hooks_info

        return result
