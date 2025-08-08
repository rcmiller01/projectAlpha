#!/usr/bin/env python3
"""
Meta-Watcher - Monitor Mirror Responsiveness

Enhanced with security features:
- Meta watcher security with monitoring validation
- Session management and access control for watcher operations
- Rate limiting and monitoring for system health checks
- Comprehensive audit logging for meta-monitoring activities

This module implements a meta-watcher that monitors Mirror's responsiveness
and triggers Anchor intervention if Mirror becomes unresponsive.
"""

import logging
import time
import threading
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Tuple
from collections import deque, defaultdict
from enum import Enum

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

class MirrorState(Enum):
    """Mirror responsiveness states"""
    RESPONSIVE = "responsive"
    SLOW = "slow"
    UNRESPONSIVE = "unresponsive"
    UNKNOWN = "unknown"

# Security configuration
META_SESSION_LENGTH = 32
MAX_RESPONSE_TIMEOUT = 300  # 5 minutes max
MAX_CHECK_INTERVAL = 3600   # 1 hour max
META_RATE_LIMIT = 50        # meta operations per hour per session
MAX_RESPONSE_HISTORY = 1000

# Thread safety
meta_lock = threading.Lock()

# Session management
meta_sessions = {}
session_expiry_hours = 24

# Rate limiting
meta_requests = defaultdict(lambda: deque())

# Access monitoring
meta_access_history = deque(maxlen=1000)

def generate_meta_session() -> str:
    """Generate a secure meta watcher session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"meta:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:META_SESSION_LENGTH]

def validate_meta_session(session_token: str) -> bool:
    """Validate meta watcher session token"""
    if not session_token or len(session_token) != META_SESSION_LENGTH:
        return False

    if session_token not in meta_sessions:
        return False

    # Check if session has expired
    session_data = meta_sessions[session_token]
    if datetime.now() > session_data['expires_at']:
        del meta_sessions[session_token]
        return False

    # Update last access time
    session_data['last_access'] = datetime.now()
    return True

def check_meta_rate_limit(session_token: str) -> bool:
    """Check if meta operation rate limit is exceeded"""
    current_time = time.time()

    # Clean old requests
    while (meta_requests[session_token] and
           meta_requests[session_token][0] < current_time - 3600):  # 1 hour window
        meta_requests[session_token].popleft()

    # Check limit
    if len(meta_requests[session_token]) >= META_RATE_LIMIT:
        logger.warning(f"Meta rate limit exceeded for session: {session_token[:8]}...")
        return False

    # Add current request
    meta_requests[session_token].append(current_time)
    return True

def validate_watcher_config(response_timeout: int, unresponsive_cycles: int, check_interval: int) -> Tuple[bool, str]:
    """Validate meta watcher configuration"""
    try:
        # Validate response timeout
        if not isinstance(response_timeout, int) or response_timeout <= 0:
            return False, "Response timeout must be a positive integer"

        if response_timeout > MAX_RESPONSE_TIMEOUT:
            return False, f"Response timeout too large (max {MAX_RESPONSE_TIMEOUT} seconds)"

        # Validate unresponsive cycles
        if not isinstance(unresponsive_cycles, int) or unresponsive_cycles <= 0:
            return False, "Unresponsive cycles must be a positive integer"

        if unresponsive_cycles > 100:
            return False, "Unresponsive cycles too large (max 100)"

        # Validate check interval
        if not isinstance(check_interval, int) or check_interval <= 0:
            return False, "Check interval must be a positive integer"

        if check_interval > MAX_CHECK_INTERVAL:
            return False, f"Check interval too large (max {MAX_CHECK_INTERVAL} seconds)"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating watcher config: {str(e)}")
        return False, f"Validation error: {str(e)}"

def log_meta_activity(activity_type: str, session_token: str, details: Dict[str, Any], status: str = "success"):
    """Log meta watcher access activities"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'session': session_token[:8] + "..." if session_token else "none",
            'details': details,
            'status': status,
            'thread_id': threading.get_ident()
        }

        meta_access_history.append(log_entry)

        logger.info(f"Meta access logged: {activity_type} ({status})")

        if status != "success":
            logger.warning(f"Meta access issue: {activity_type} failed with {status}")

    except Exception as e:
        logger.error(f"Error logging meta access: {str(e)}")

class MetaWatcher:
    """
    Meta-watcher system to monitor Mirror responsiveness with enhanced security.

    Features:
    - Mirror response monitoring with validation
    - Session-based authentication and authorization
    - Rate limiting and access monitoring
    - Comprehensive audit logging
    - Thread-safe concurrent monitoring

    Tracks Mirror's response times and health status,
    triggering Anchor intervention when necessary.
    """

    def __init__(self,
                 response_timeout: int = 30,
                 unresponsive_cycles: int = 3,
                 check_interval: int = 10,
                 session_token: Optional[str] = None):
        """
        Initialize the MetaWatcher with security features.

        Args:
            response_timeout (int): Seconds to wait for Mirror response
            unresponsive_cycles (int): Cycles before considering Mirror unresponsive
            check_interval (int): Seconds between health checks
            session_token (str): Session token for authentication
        """
        # Validate configuration
        is_valid, validation_message = validate_watcher_config(response_timeout, unresponsive_cycles, check_interval)
        if not is_valid:
            raise ValueError(f"Invalid watcher configuration: {validation_message}")

        self.session_token = session_token or self.create_session()
        self.creation_time = datetime.now()

        # Clamp values to safe ranges
        self.response_timeout = min(response_timeout, MAX_RESPONSE_TIMEOUT)
        self.unresponsive_cycles = min(unresponsive_cycles, 100)
        self.check_interval = min(check_interval, MAX_CHECK_INTERVAL)

        self.mirror_state = MirrorState.UNKNOWN
        self.last_response_time = None
        self.failed_cycles = 0
        self.response_history = deque(maxlen=MAX_RESPONSE_HISTORY)
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitoring_count = 0

        # Callback for when Anchor needs to be fired
        self.anchor_callback: Optional[Callable] = None

        log_meta_activity("initialization", self.session_token, {
            "response_timeout": self.response_timeout,
            "unresponsive_cycles": self.unresponsive_cycles,
            "check_interval": self.check_interval,
            "creation_time": self.creation_time.isoformat()
        })

        logger.info(f"MetaWatcher initialized with security features")

    def create_session(self) -> str:
        """Create a new meta watcher session"""
        with meta_lock:
            session_token = generate_meta_session()
            meta_sessions[session_token] = {
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=session_expiry_hours),
                'last_access': datetime.now(),
                'meta_operations': 0
            }
            return session_token

    def validate_session(self, session_token: Optional[str] = None) -> bool:
        """Validate session for meta watcher operations"""
        token_to_validate = session_token or self.session_token

        if not token_to_validate:
            logger.warning("No session token provided for meta validation")
            return False

        return validate_meta_session(token_to_validate)

    def start_monitoring(self, session_token: Optional[str] = None) -> bool:
        """
        Start the monitoring thread with security validation.

        Args:
            session_token: Session token for authentication

        Returns:
            Success status
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_meta_activity("start_monitoring", session_token or self.session_token,
                                 {"status": "session_invalid"}, "failed")
                return False

            current_token = session_token or self.session_token

            # Check rate limit
            if not check_meta_rate_limit(current_token):
                log_meta_activity("start_monitoring", current_token,
                                 {"status": "rate_limited"}, "failed")
                return False

            with meta_lock:
                if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("MetaWatcher monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("MetaWatcher monitoring stopped")

    def set_anchor_callback(self, callback: Callable):
        """
        Set the callback function to fire Anchor.

        Args:
            callback (Callable): Function to call when Anchor needs to be fired
        """
        self.anchor_callback = callback
        logger.info("Anchor callback registered with MetaWatcher")

    def record_mirror_response(self, response_time: float, success: bool = True):
        """
        Record a Mirror response for monitoring.

        Args:
            response_time (float): Time taken for Mirror to respond
            success (bool): Whether the response was successful
        """
        current_time = time.time()

        self.response_history.append({
            "timestamp": current_time,
            "response_time": response_time,
            "success": success
        })

        # Keep only recent history (last 100 responses)
        if len(self.response_history) > 100:
            self.response_history = self.response_history[-100:]

        if success:
            self.last_response_time = current_time
            self.failed_cycles = 0
            self._update_mirror_state(response_time)
        else:
            self.failed_cycles += 1

        logger.debug(f"Mirror response recorded: {response_time:.2f}s, success: {success}")

    def ping_mirror(self) -> bool:
        """
        Ping Mirror to check responsiveness.

        Returns:
            bool: True if Mirror responds, False otherwise
        """
        try:
            start_time = time.time()

            # TODO: Replace with actual Mirror ping implementation
            # For now, simulate a ping with random response
            import random
            time.sleep(random.uniform(0.1, 2.0))  # Simulate response time
            success = random.random() > 0.1  # 90% success rate

            response_time = time.time() - start_time
            self.record_mirror_response(response_time, success)

            return success

        except Exception as e:
            logger.error(f"Failed to ping Mirror: {e}")
            self.record_mirror_response(self.response_timeout, False)
            return False

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check Mirror responsiveness
                if not self.ping_mirror():
                    self.failed_cycles += 1
                    logger.warning(f"Mirror ping failed. Failed cycles: {self.failed_cycles}")

                # Check if Mirror is unresponsive
                if self.failed_cycles >= self.unresponsive_cycles:
                    self._handle_unresponsive_mirror()

                # Wait before next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in MetaWatcher monitor loop: {e}")
                time.sleep(self.check_interval)

    def _update_mirror_state(self, response_time: float):
        """
        Update Mirror state based on response time.

        Args:
            response_time (float): Time taken for Mirror to respond
        """
        if response_time < 5.0:
            self.mirror_state = MirrorState.RESPONSIVE
        elif response_time < 15.0:
            self.mirror_state = MirrorState.SLOW
        else:
            self.mirror_state = MirrorState.UNRESPONSIVE

    def _handle_unresponsive_mirror(self):
        """Handle unresponsive Mirror by firing Anchor."""
        logger.critical(f"Mirror unresponsive for {self.failed_cycles} cycles. Firing Anchor!")

        self.mirror_state = MirrorState.UNRESPONSIVE

        # Fire Anchor callback if available
        if self.anchor_callback:
            try:
                self.anchor_callback({
                    "reason": "mirror_unresponsive",
                    "failed_cycles": self.failed_cycles,
                    "last_response": self.last_response_time,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.error(f"Failed to fire Anchor callback: {e}")
        else:
            logger.error("No Anchor callback registered!")

    def get_mirror_status(self) -> Dict[str, Any]:
        """
        Get current Mirror status.

        Returns:
            Dict: Mirror status information
        """
        recent_responses = self.response_history[-10:] if self.response_history else []
        avg_response_time = (
            sum(r["response_time"] for r in recent_responses) / len(recent_responses)
            if recent_responses else None
        )

        return {
            "state": self.mirror_state.value,
            "failed_cycles": self.failed_cycles,
            "last_response_time": self.last_response_time,
            "average_response_time": avg_response_time,
            "total_responses": len(self.response_history),
            "is_monitoring": self.is_monitoring
        }

    def reset_monitoring(self):
        """Reset monitoring state."""
        self.failed_cycles = 0
        self.mirror_state = MirrorState.UNKNOWN
        self.response_history.clear()
        logger.info("MetaWatcher monitoring state reset")

# Global meta-watcher instance
meta_watcher = MetaWatcher()

def fire_anchor_intervention(reason_data: Dict[str, Any]):
    """
    Default Anchor intervention callback.

    Args:
        reason_data (Dict): Data about why Anchor was fired
    """
    logger.critical(f"🔥 ANCHOR INTERVENTION FIRED: {reason_data}")

    # TODO: Implement actual Anchor intervention
    # This could include:
    # - Stopping autopilot processes
    # - Alerting administrators
    # - Switching to safe mode
    # - Rolling back recent changes

# Set default callback
meta_watcher.set_anchor_callback(fire_anchor_intervention)

if __name__ == "__main__":
    # Test the meta-watcher
    print("Starting MetaWatcher test...")

    meta_watcher.start_monitoring()

    try:
        # Let it run for a bit
        time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping test...")
    finally:
        meta_watcher.stop_monitoring()
        print("MetaWatcher test completed")
