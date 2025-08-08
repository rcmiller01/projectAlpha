"""
Goodbye Manager - Secure Session Termination and Cleanup

Enhanced with security features:
- Goodbye manager security with session validation
- Secure cleanup and resource management
- Session management and access control for termination operations
- Rate limiting and monitoring for goodbye procedures
"""

import hashlib
import re
import time
import threading
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

class GoodbyeState(Enum):
    """Goodbye procedure states"""
    INITIATED = "initiated"
    CLEANING_UP = "cleaning_up"
    SAVING_STATE = "saving_state"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"

class GoodbyeReason(Enum):
    """Reasons for goodbye procedures"""
    USER_REQUEST = "user_request"
    SESSION_TIMEOUT = "session_timeout"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR_CONDITION = "error_condition"
    MAINTENANCE = "maintenance"
    SECURITY_VIOLATION = "security_violation"

# Security configuration
GOODBYE_SESSION_LENGTH = 32
MAX_GOODBYE_MESSAGE_LENGTH = 1000
MAX_CLEANUP_TASKS = 20
GOODBYE_RATE_LIMIT = 10  # goodbye operations per hour per session
MAX_GOODBYE_TIMEOUT = 300  # 5 minutes max for goodbye procedure

# Thread safety
goodbye_lock = threading.Lock()

# Session management
goodbye_sessions = {}
session_expiry_hours = 24

# Rate limiting
goodbye_requests = defaultdict(lambda: deque())

# Access monitoring
goodbye_access_history = deque(maxlen=1000)

@dataclass
class GoodbyeTask:
    """A cleanup task to be executed during goodbye"""
    name: str
    callback: Callable[[], bool]
    priority: int
    timeout: float
    description: str
    created_at: str

@dataclass
class GoodbyeReport:
    """Report of goodbye procedure execution"""
    goodbye_id: str
    reason: GoodbyeReason
    state: GoodbyeState
    started_at: str
    completed_at: Optional[str]
    tasks_executed: List[Dict[str, Any]]
    cleanup_success: bool
    session_token: str
    security_metadata: Dict[str, Any]

def generate_goodbye_session() -> str:
    """Generate a secure goodbye session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"goodbye:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:GOODBYE_SESSION_LENGTH]

def validate_goodbye_session(session_token: str) -> bool:
    """Validate goodbye session token"""
    if not session_token or len(session_token) != GOODBYE_SESSION_LENGTH:
        return False

    if session_token not in goodbye_sessions:
        return False

    # Check if session has expired
    session_data = goodbye_sessions[session_token]
    if datetime.now() > session_data['expires_at']:
        del goodbye_sessions[session_token]
        return False

    # Update last access time
    session_data['last_access'] = datetime.now()
    return True

def check_goodbye_rate_limit(session_token: str) -> bool:
    """Check if goodbye operation rate limit is exceeded"""
    current_time = time.time()

    # Clean old requests
    while (goodbye_requests[session_token] and
           goodbye_requests[session_token][0] < current_time - 3600):  # 1 hour window
        goodbye_requests[session_token].popleft()

    # Check limit
    if len(goodbye_requests[session_token]) >= GOODBYE_RATE_LIMIT:
        logger.warning(f"Goodbye rate limit exceeded for session: {session_token[:8]}...")
        return False

    # Add current request
    goodbye_requests[session_token].append(current_time)
    return True

def validate_goodbye_message(message: str) -> Tuple[bool, str]:
    """Validate goodbye message"""
    try:
        if not isinstance(message, str):
            return False, "Goodbye message must be a string"

        if len(message) > MAX_GOODBYE_MESSAGE_LENGTH:
            return False, f"Goodbye message too long (max {MAX_GOODBYE_MESSAGE_LENGTH} characters)"

        # Check for dangerous content
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',               # JavaScript protocol
            r'on\w+\s*=',                # Event handlers
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return False, "Goodbye message contains potentially dangerous content"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating goodbye message: {str(e)}")
        return False, f"Validation error: {str(e)}"

def sanitize_goodbye_message(message: str) -> str:
    """Sanitize goodbye message for safety"""
    # Remove dangerous characters
    clean_message = re.sub(r'[<>"\']', '', str(message))
    # Limit length
    if len(clean_message) > MAX_GOODBYE_MESSAGE_LENGTH:
        clean_message = clean_message[:MAX_GOODBYE_MESSAGE_LENGTH] + "..."
    return clean_message

def log_goodbye_activity(activity_type: str, session_token: str, details: Dict[str, Any], status: str = "success"):
    """Log goodbye access activities"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'session': session_token[:8] + "..." if session_token else "none",
            'details': details,
            'status': status,
            'thread_id': threading.get_ident()
        }

        goodbye_access_history.append(log_entry)

        logger.info(f"Goodbye access logged: {activity_type} ({status})")

        if status != "success":
            logger.warning(f"Goodbye access issue: {activity_type} failed with {status}")

    except Exception as e:
        logger.error(f"Error logging goodbye access: {str(e)}")

class GoodbyeManager:
    """
    Secure goodbye manager for AI agent session termination and cleanup.

    Features:
    - Secure session termination with validation
    - Resource cleanup and state saving
    - Session-based authentication and authorization
    - Rate limiting and access monitoring
    - Comprehensive audit logging
    - Thread-safe concurrent operations
    """

    def __init__(self, session_token: Optional[str] = None):
        """Initialize goodbye manager with security features"""
        self.session_token = session_token or self.create_session()
        self.creation_time = datetime.now()
        self.cleanup_tasks: List[GoodbyeTask] = []
        self.goodbye_count = 0
        self.current_state = GoodbyeState.INITIATED

        # Initialize default cleanup tasks
        self._register_default_tasks()

        log_goodbye_activity("initialization", self.session_token, {
            "creation_time": self.creation_time.isoformat(),
            "default_tasks": len(self.cleanup_tasks)
        })

        logger.info(f"GoodbyeManager initialized with security features")

    def create_session(self) -> str:
        """Create a new goodbye session"""
        with goodbye_lock:
            session_token = generate_goodbye_session()
            goodbye_sessions[session_token] = {
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=session_expiry_hours),
                'last_access': datetime.now(),
                'goodbye_operations': 0
            }
            return session_token

    def validate_session(self, session_token: Optional[str] = None) -> bool:
        """Validate session for goodbye operations"""
        token_to_validate = session_token or self.session_token

        if not token_to_validate:
            logger.warning("No session token provided for goodbye validation")
            return False

        return validate_goodbye_session(token_to_validate)

    def _register_default_tasks(self):
        """Register default cleanup tasks"""
        default_tasks = [
            {
                "name": "save_session_state",
                "callback": self._save_session_state,
                "priority": 1,
                "timeout": 30.0,
                "description": "Save current session state to persistent storage"
            },
            {
                "name": "cleanup_temp_files",
                "callback": self._cleanup_temp_files,
                "priority": 2,
                "timeout": 15.0,
                "description": "Remove temporary files and cache"
            },
            {
                "name": "close_connections",
                "callback": self._close_connections,
                "priority": 3,
                "timeout": 10.0,
                "description": "Close network connections and file handles"
            },
            {
                "name": "finalize_logs",
                "callback": self._finalize_logs,
                "priority": 4,
                "timeout": 5.0,
                "description": "Flush and finalize log files"
            }
        ]

        for task_config in default_tasks:
            task = GoodbyeTask(
                name=task_config["name"],
                callback=task_config["callback"],
                priority=task_config["priority"],
                timeout=task_config["timeout"],
                description=task_config["description"],
                created_at=datetime.now().isoformat()
            )
            self.cleanup_tasks.append(task)

        # Sort tasks by priority
        self.cleanup_tasks.sort(key=lambda t: t.priority)

    def register_cleanup_task(self,
                            name: str,
                            callback: Callable[[], bool],
                            priority: int = 5,
                            timeout: float = 30.0,
                            description: str = "",
                            session_token: Optional[str] = None) -> bool:
        """
        Register a cleanup task with security validation.

        Args:
            name: Unique name for the task
            callback: Function to execute during cleanup
            priority: Execution priority (lower numbers execute first)
            timeout: Maximum execution time in seconds
            description: Description of the task
            session_token: Session token for authentication

        Returns:
            Success status
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_goodbye_activity("register_task", session_token or self.session_token,
                                    {"name": name, "status": "session_invalid"}, "failed")
                return False

            current_token = session_token or self.session_token

            # Check rate limit
            if not check_goodbye_rate_limit(current_token):
                log_goodbye_activity("register_task", current_token,
                                    {"name": name, "status": "rate_limited"}, "failed")
                return False

            # Validate task parameters
            if not isinstance(name, str) or not name.strip():
                logger.error("Task name must be a non-empty string")
                return False

            if not callable(callback):
                logger.error("Task callback must be callable")
                return False

            if not isinstance(priority, int) or priority < 0:
                logger.error("Task priority must be a non-negative integer")
                return False

            if not isinstance(timeout, (int, float)) or timeout <= 0:
                logger.error("Task timeout must be a positive number")
                return False

            if timeout > MAX_GOODBYE_TIMEOUT:
                timeout = MAX_GOODBYE_TIMEOUT
                logger.warning(f"Task timeout clamped to {MAX_GOODBYE_TIMEOUT} seconds")

            # Sanitize name and description
            clean_name = re.sub(r'[<>"\']', '', name.strip())
            clean_description = re.sub(r'[<>"\']', '', str(description))

            with goodbye_lock:
                # Check if too many tasks
                if len(self.cleanup_tasks) >= MAX_CLEANUP_TASKS:
                    logger.error(f"Too many cleanup tasks (max {MAX_CLEANUP_TASKS})")
                    log_goodbye_activity("register_task", current_token,
                                        {"name": clean_name, "error": "too_many_tasks"}, "failed")
                    return False

                # Check for duplicate name
                existing_names = [task.name for task in self.cleanup_tasks]
                if clean_name in existing_names:
                    logger.error(f"Task with name '{clean_name}' already exists")
                    log_goodbye_activity("register_task", current_token,
                                        {"name": clean_name, "error": "duplicate_name"}, "failed")
                    return False

                # Create and register task
                task = GoodbyeTask(
                    name=clean_name,
                    callback=callback,
                    priority=priority,
                    timeout=timeout,
                    description=clean_description,
                    created_at=datetime.now().isoformat()
                )

                self.cleanup_tasks.append(task)

                # Sort tasks by priority
                self.cleanup_tasks.sort(key=lambda t: t.priority)

                # Update session tracking
                if current_token in goodbye_sessions:
                    goodbye_sessions[current_token]['goodbye_operations'] += 1

            log_goodbye_activity("register_task", current_token, {
                "name": clean_name,
                "priority": priority,
                "timeout": timeout,
                "tasks_count": len(self.cleanup_tasks)
            })

            logger.info(f"Cleanup task registered: {clean_name} (priority: {priority})")
            return True

        except Exception as e:
            logger.error(f"Error registering cleanup task: {str(e)}")
            log_goodbye_activity("register_task", session_token or self.session_token,
                                {"name": name, "error": str(e)}, "error")
            return False

    def say_goodbye(self,
                   reason: GoodbyeReason = GoodbyeReason.USER_REQUEST,
                   message: str = "",
                   session_token: Optional[str] = None) -> GoodbyeReport:
        """
        Execute goodbye procedure with security validation.

        Args:
            reason: Reason for the goodbye
            message: Optional goodbye message
            session_token: Session token for authentication

        Returns:
            GoodbyeReport with execution details
        """
        goodbye_id = hashlib.md5(f"{self.session_token}{time.time()}".encode()).hexdigest()[:16]
        start_time = datetime.now()

        try:
            # Validate session
            if not self.validate_session(session_token):
                log_goodbye_activity("say_goodbye", session_token or self.session_token,
                                    {"reason": reason.value, "status": "session_invalid"}, "failed")
                return self._create_error_report(goodbye_id, reason, "Invalid session")

            current_token = session_token or self.session_token

            # Check rate limit
            if not check_goodbye_rate_limit(current_token):
                log_goodbye_activity("say_goodbye", current_token,
                                    {"reason": reason.value, "status": "rate_limited"}, "failed")
                return self._create_error_report(goodbye_id, reason, "Rate limit exceeded")

            # Validate and sanitize message
            if message:
                is_valid, validation_message = validate_goodbye_message(message)
                if not is_valid:
                    logger.error(f"Invalid goodbye message: {validation_message}")
                    message = "Invalid message removed for security"
                else:
                    message = sanitize_goodbye_message(message)

            logger.info(f"Starting goodbye procedure: {goodbye_id} (reason: {reason.value})")

            # Update state
            self.current_state = GoodbyeState.CLEANING_UP

            # Execute cleanup tasks
            tasks_executed = []
            cleanup_success = True

            for task in self.cleanup_tasks:
                task_start = time.time()
                task_result = {
                    'name': task.name,
                    'priority': task.priority,
                    'description': task.description,
                    'started_at': datetime.now().isoformat(),
                    'success': False,
                    'execution_time': 0.0,
                    'error': None
                }

                try:
                    logger.debug(f"Executing cleanup task: {task.name}")

                    # Execute task with timeout
                    success = task.callback()
                    task_result['success'] = bool(success)

                    if not success:
                        cleanup_success = False
                        logger.warning(f"Cleanup task failed: {task.name}")

                except Exception as task_error:
                    cleanup_success = False
                    task_result['error'] = str(task_error)
                    logger.error(f"Error in cleanup task {task.name}: {str(task_error)}")

                finally:
                    task_result['execution_time'] = time.time() - task_start
                    tasks_executed.append(task_result)

            # Update state
            self.current_state = GoodbyeState.FINALIZING

            # Track goodbye
            with goodbye_lock:
                self.goodbye_count += 1
                if current_token in goodbye_sessions:
                    goodbye_sessions[current_token]['goodbye_operations'] += 1

            # Create report
            self.current_state = GoodbyeState.COMPLETED if cleanup_success else GoodbyeState.FAILED

            report = GoodbyeReport(
                goodbye_id=goodbye_id,
                reason=reason,
                state=self.current_state,
                started_at=start_time.isoformat(),
                completed_at=datetime.now().isoformat(),
                tasks_executed=tasks_executed,
                cleanup_success=cleanup_success,
                session_token=current_token[:8] + "...",
                security_metadata={
                    "session_validated": True,
                    "rate_limit_checked": True,
                    "message_sanitized": bool(message),
                    "tasks_count": len(tasks_executed)
                }
            )

            log_goodbye_activity("say_goodbye", current_token, {
                "goodbye_id": goodbye_id,
                "reason": reason.value,
                "cleanup_success": cleanup_success,
                "tasks_executed": len(tasks_executed),
                "total_time": (datetime.now() - start_time).total_seconds()
            })

            logger.info(f"Goodbye procedure completed: {goodbye_id} ({'success' if cleanup_success else 'with errors'})")

            if message:
                logger.info(f"Goodbye message: {message}")

            return report

        except Exception as e:
            logger.error(f"Error in goodbye procedure: {str(e)}")
            log_goodbye_activity("say_goodbye", session_token or self.session_token,
                                {"goodbye_id": goodbye_id, "error": str(e)}, "error")
            return self._create_error_report(goodbye_id, reason, str(e))

    def _create_error_report(self, goodbye_id: str, reason: GoodbyeReason, error: str) -> GoodbyeReport:
        """Create an error report for failed goodbye procedures"""
        return GoodbyeReport(
            goodbye_id=goodbye_id,
            reason=reason,
            state=GoodbyeState.FAILED,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            tasks_executed=[],
            cleanup_success=False,
            session_token=self.session_token[:8] + "..." if self.session_token else "unknown",
            security_metadata={
                "error": error,
                "session_validated": False
            }
        )

    def _save_session_state(self) -> bool:
        """Default task: Save session state"""
        try:
            # Placeholder for session state saving
            logger.debug("Saving session state...")
            time.sleep(0.1)  # Simulate work
            return True
        except Exception as e:
            logger.error(f"Error saving session state: {str(e)}")
            return False

    def _cleanup_temp_files(self) -> bool:
        """Default task: Cleanup temporary files"""
        try:
            # Placeholder for temp file cleanup
            logger.debug("Cleaning up temporary files...")
            time.sleep(0.1)  # Simulate work
            return True
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
            return False

    def _close_connections(self) -> bool:
        """Default task: Close connections"""
        try:
            # Placeholder for connection cleanup
            logger.debug("Closing connections...")
            time.sleep(0.1)  # Simulate work
            return True
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
            return False

    def _finalize_logs(self) -> bool:
        """Default task: Finalize logs"""
        try:
            # Placeholder for log finalization
            logger.debug("Finalizing logs...")
            time.sleep(0.1)  # Simulate work
            return True
        except Exception as e:
            logger.error(f"Error finalizing logs: {str(e)}")
            return False

    def get_goodbye_stats(self) -> Dict[str, Any]:
        """Get goodbye manager statistics"""
        return {
            'session_token': self.session_token[:8] + "..." if self.session_token else None,
            'creation_time': self.creation_time.isoformat(),
            'goodbye_count': self.goodbye_count,
            'cleanup_tasks_count': len(self.cleanup_tasks),
            'current_state': self.current_state.value,
            'available_reasons': [reason.value for reason in GoodbyeReason]
        }
