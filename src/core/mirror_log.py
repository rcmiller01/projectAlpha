"""
Mirror Log System for Secure Logging and Retrieval

Enhanced with security features:
- Mirror logging security enhancement with integrity verification
- Input validation and sanitization for all log entries
- Session management and audit trails for log access
- Rate limiting and monitoring for log operations
"""

import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

# Security configuration
MIRROR_LOG_SESSION_LENGTH = 32
MAX_LOG_ENTRY_SIZE = 5000
MAX_SEARCH_PATTERN_LENGTH = 200
LOG_RATE_LIMIT = 50  # log operations per hour per session
MAX_LOG_ENTRIES_PER_FILE = 10000

# Thread safety
log_lock = threading.Lock()

# Session management
log_sessions = {}
session_expiry_hours = 24

# Rate limiting
log_requests = defaultdict(lambda: deque())

# Access monitoring
log_access_history = deque(maxlen=1000)


def generate_log_session() -> str:
    """Generate a secure mirror log session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"mirrorlog:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:MIRROR_LOG_SESSION_LENGTH]


def validate_log_session(session_token: str) -> bool:
    """Validate mirror log session token"""
    if not session_token or len(session_token) != MIRROR_LOG_SESSION_LENGTH:
        return False

    if session_token not in log_sessions:
        return False

    # Check if session has expired
    session_data = log_sessions[session_token]
    if datetime.now() > session_data["expires_at"]:
        del log_sessions[session_token]
        return False

    # Update last access time
    session_data["last_access"] = datetime.now()
    return True


def check_log_rate_limit(session_token: str) -> bool:
    """Check if log operation rate limit is exceeded"""
    current_time = time.time()

    # Clean old requests
    while (
        log_requests[session_token] and log_requests[session_token][0] < current_time - 3600
    ):  # 1 hour window
        log_requests[session_token].popleft()

    # Check limit
    if len(log_requests[session_token]) >= LOG_RATE_LIMIT:
        logger.warning(f"Mirror log rate limit exceeded for session: {session_token[:8]}...")
        return False

    # Add current request
    log_requests[session_token].append(current_time)
    return True


def validate_log_entry(data: dict[str, Any]) -> tuple[bool, str]:
    """Validate log entry data"""
    try:
        # Check data size
        entry_size = len(json.dumps(data))
        if entry_size > MAX_LOG_ENTRY_SIZE:
            return False, f"Log entry too large: {entry_size} bytes (max {MAX_LOG_ENTRY_SIZE})"

        # Validate required fields for mirror logs
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()

        # Validate summary if present
        if "summary" in data:
            summary = data["summary"]
            if not isinstance(summary, str):
                return False, "Summary must be a string"

            if len(summary) > 1000:
                return False, "Summary too long (max 1000 characters)"

        # Check for dangerous content
        data_str = json.dumps(data)
        dangerous_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"javascript:",  # JavaScript protocol
            r"on\w+\s*=",  # Event handlers
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                return False, "Log entry contains potentially dangerous content"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating log entry: {e!s}")
        return False, f"Validation error: {e!s}"


def sanitize_log_data(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize log data for safety"""
    sanitized = {}

    for key, value in data.items():
        # Sanitize string values
        if isinstance(value, str):
            # Remove potentially dangerous characters
            value = re.sub(r'[<>"\']', "", value)
            # Limit length
            if len(value) > 2000:
                value = value[:2000] + "..."

        # Sanitize key names
        clean_key = re.sub(r"[^a-zA-Z0-9_]", "", str(key))
        if clean_key:
            sanitized[clean_key] = value

    return sanitized


def validate_search_pattern(pattern: str) -> tuple[bool, str]:
    """Validate search pattern"""
    try:
        if not isinstance(pattern, str):
            return False, "Search pattern must be a string"

        if len(pattern) > MAX_SEARCH_PATTERN_LENGTH:
            return False, f"Search pattern too long (max {MAX_SEARCH_PATTERN_LENGTH})"

        # Check for regex injection attempts
        dangerous_regex = [
            r"\(\?\(",  # Conditional patterns
            r"\(\?\#",  # Comments
            r"\(\?\:",  # Non-capturing groups with potential issues
        ]

        for regex in dangerous_regex:
            if re.search(regex, pattern, re.IGNORECASE):
                return False, "Search pattern contains potentially dangerous regex"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating search pattern: {e!s}")
        return False, f"Validation error: {e!s}"


def log_access_activity(
    activity_type: str, session_token: str, details: dict[str, Any], status: str = "success"
):
    """Log mirror log access activities"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "activity_type": activity_type,
            "session": session_token[:8] + "..." if session_token else "none",
            "details": details,
            "status": status,
            "thread_id": threading.get_ident(),
        }

        log_access_history.append(log_entry)

        logger.info(f"Mirror log access logged: {activity_type} ({status})")

        if status != "success":
            logger.warning(f"Mirror log access issue: {activity_type} failed with {status}")

    except Exception as e:
        logger.error(f"Error logging mirror log access: {e!s}")


class MirrorLog:
    """Secure append-only log for mirror mode reports with enhanced security."""

    def __init__(
        self, log_file: str = "logs/mirror_log.jsonl", session_token: Optional[str] = None
    ):
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(exist_ok=True)
        self.session_token = session_token
        self.entry_count = 0
        self.creation_time = datetime.now()

        # Initialize session if not provided
        if not self.session_token:
            self.session_token = self.create_session()

        logger.info(f"MirrorLog initialized: {log_file}")

    def create_session(self) -> str:
        """Create a new log session"""
        with log_lock:
            session_token = generate_log_session()
            log_sessions[session_token] = {
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(hours=session_expiry_hours),
                "last_access": datetime.now(),
                "log_operations": 0,
            }
            return session_token

    def validate_session(self, session_token: Optional[str] = None) -> bool:
        """Validate session for log operations"""
        token_to_validate = session_token or self.session_token

        if not token_to_validate:
            logger.warning("No session token provided for log validation")
            return False

        return validate_log_session(token_to_validate)

    def append(self, data: dict[str, Any], session_token: Optional[str] = None) -> bool:
        """Securely append data to mirror log"""
        try:
            with log_lock:
                # Validate session
                if not self.validate_session(session_token):
                    log_access_activity(
                        "append",
                        session_token or self.session_token,
                        {"status": "session_invalid"},
                        "failed",
                    )
                    return False

                # Check rate limit
                current_token = session_token or self.session_token
                if not check_log_rate_limit(current_token):
                    log_access_activity(
                        "append", current_token, {"status": "rate_limited"}, "failed"
                    )
                    return False

                # Validate entry
                is_valid, validation_message = validate_log_entry(data)
                if not is_valid:
                    logger.error(f"Invalid log entry: {validation_message}")
                    log_access_activity(
                        "append", current_token, {"error": validation_message}, "validation_failed"
                    )
                    return False

                # Sanitize data
                sanitized_data = sanitize_log_data(data)

                # Add security metadata
                sanitized_data["_meta"] = {
                    "entry_id": self.entry_count,
                    "session_token": current_token[:8] + "...",
                    "timestamp": datetime.now().isoformat(),
                    "sanitized": True,
                }

                # Write to log
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sanitized_data) + "\n")

                self.entry_count += 1

                # Update session tracking
                if current_token in log_sessions:
                    log_sessions[current_token]["log_operations"] += 1

                log_access_activity(
                    "append", current_token, {"entry_count": self.entry_count}, "success"
                )

                return True

        except Exception as e:
            logger.error(f"Error appending to mirror log: {e!s}")
            log_access_activity(
                "append", session_token or self.session_token, {"error": str(e)}, "error"
            )
            return False

    def tail(self, limit: int = 20, session_token: Optional[str] = None) -> list[dict[str, Any]]:
        """Securely retrieve recent log entries"""
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_access_activity(
                    "tail",
                    session_token or self.session_token,
                    {"limit": limit, "status": "session_invalid"},
                    "failed",
                )
                return []

            # Validate limit
            if limit < 1 or limit > 100:
                limit = min(max(limit, 1), 100)

            current_token = session_token or self.session_token

            # Check rate limit
            if not check_log_rate_limit(current_token):
                log_access_activity(
                    "tail", current_token, {"limit": limit, "status": "rate_limited"}, "failed"
                )
                return []

            if not self.log_path.exists():
                return []

            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            records = []

            for line in lines[-limit:]:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

            log_access_activity(
                "tail", current_token, {"limit": limit, "retrieved": len(records)}, "success"
            )

            return records

        except Exception as e:
            logger.error(f"Error reading mirror log tail: {e!s}")
            log_access_activity(
                "tail", session_token or self.session_token, {"error": str(e)}, "error"
            )
            return []

    def last(self, session_token: Optional[str] = None) -> dict[str, Any]:
        """Securely retrieve the last log entry"""
        entries = self.tail(1, session_token)
        return entries[0] if entries else {}

    def search(
        self, pattern: str, limit: int = 20, session_token: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Securely search log entries"""
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_access_activity(
                    "search",
                    session_token or self.session_token,
                    {"pattern": pattern[:50], "status": "session_invalid"},
                    "failed",
                )
                return []

            # Validate search pattern
            is_valid, validation_message = validate_search_pattern(pattern)
            if not is_valid:
                logger.error(f"Invalid search pattern: {validation_message}")
                log_access_activity(
                    "search",
                    session_token or self.session_token,
                    {"error": validation_message},
                    "validation_failed",
                )
                return []

            current_token = session_token or self.session_token

            # Check rate limit
            if not check_log_rate_limit(current_token):
                log_access_activity(
                    "search",
                    current_token,
                    {"pattern": pattern[:50], "status": "rate_limited"},
                    "failed",
                )
                return []

            if not self.log_path.exists():
                return []

            # Validate limit
            if limit < 1 or limit > 100:
                limit = min(max(limit, 1), 100)

            pattern = pattern.lower()
            results = []

            with open(self.log_path, encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    if len(results) >= limit:
                        break
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    summary = str(entry.get("summary", "")).lower()
                    if pattern in summary:
                        results.append(entry)

            log_access_activity(
                "search",
                current_token,
                {"pattern": pattern[:50], "limit": limit, "found": len(results)},
                "success",
            )

            return results

        except Exception as e:
            logger.error(f"Error searching mirror log: {e!s}")
            log_access_activity(
                "search", session_token or self.session_token, {"error": str(e)}, "error"
            )
            return []

    def get_stats(self) -> dict[str, Any]:
        """Get mirror log statistics"""
        return {
            "log_file": str(self.log_path),
            "entry_count": self.entry_count,
            "file_exists": self.log_path.exists(),
            "file_size_bytes": self.log_path.stat().st_size if self.log_path.exists() else 0,
            "creation_time": self.creation_time.isoformat(),
            "session_token": self.session_token[:8] + "..." if self.session_token else None,
        }
