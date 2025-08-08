"""
ProjectAlpha Test Configuration and Fixtures
Provides deterministic testing capabilities and shared test fixtures.
"""

import os
import random
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--deterministic",
        action="store_true",
        default=False,
        help="Run tests in deterministic mode with fixed seeds and configurations",
    )
    parser.addoption(
        "--chaos-level",
        type=int,
        default=3,
        help="Chaos testing intensity level (1-10, default: 3)",
    )
    parser.addoption(
        "--safe-mode-tests",
        action="store_true",
        default=False,
        help="Include safe mode specific tests",
    )


def pytest_configure(config):
    """Configure pytest environment based on command line options."""
    if config.getoption("--deterministic"):
        # Set deterministic mode
        os.environ["PYTEST_DETERMINISTIC"] = "true"

        # Fix random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        os.environ["PYTHONHASHSEED"] = "42"

        # Set fixed configuration for deterministic behavior
        os.environ["DRIFT_SCALING_FACTOR"] = "0.35"
        os.environ["MAX_PENALTY_THRESHOLD"] = "0.85"
        os.environ["RETRY_MAX_ATTEMPTS"] = "3"
        os.environ["RETRY_BASE_DELAY"] = "0.1"  # Faster for tests
        os.environ["RETRY_MAX_DELAY"] = "1.0"  # Faster for tests

        print("\n🎯 Deterministic test mode activated")
        print("   - Fixed random seeds (42)")
        print("   - Fixed drift scaling (0.35)")
        print("   - Fast retry configuration")


@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Set up test environment with proper isolation."""
    # Store original environment
    original_env = dict(os.environ)

    # Set test-specific environment
    test_env = {
        "TESTING": "true",
        "LOG_LEVEL": "WARNING",  # Reduce noise in tests
        "MEMORY_QUOTA_IDENTITY": "10",  # Small quotas for fast testing
        "MEMORY_QUOTA_BELIEFS": "50",
        "MEMORY_QUOTA_EPHEMERAL": "100",
        "IDEMPOTENCY_CACHE_TTL": "60",  # Short TTL for tests
        "HEALTH_CHECK_TIMEOUT": "1.0",  # Fast health checks
    }

    for key, value in test_env.items():
        os.environ[key] = value

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def deterministic_mode(request):
    """Force deterministic behavior for specific tests."""
    is_deterministic = request.config.getoption("--deterministic")

    if is_deterministic:
        # Additional deterministic setup for individual tests
        random.seed(42)
        np.random.seed(42)

    return is_deterministic


@pytest.fixture
def safe_mode_disabled():
    """Ensure safe mode is disabled unless explicitly enabled by test."""
    original_safe_mode = os.environ.get("SAFE_MODE_FORCE", "false")

    # Force safe mode off
    os.environ["SAFE_MODE_FORCE"] = "false"

    yield

    # Restore original value
    os.environ["SAFE_MODE_FORCE"] = original_safe_mode


@pytest.fixture
def mock_conductor():
    """Mock CoreConductor for isolated testing."""
    with patch("src.core.core_conductor.CoreConductor") as mock:
        conductor_instance = MagicMock()
        conductor_instance.safe_mode_enabled = False
        conductor_instance.emotion_loop_paused = False
        conductor_instance.writes_locked = False
        conductor_instance.watchdog_failure_count = 0

        # Mock safe mode methods
        conductor_instance.enter_safe_mode.return_value = True
        conductor_instance.exit_safe_mode.return_value = (True, "Exited successfully")
        conductor_instance.get_safe_mode_status.return_value = {
            "safe_mode_enabled": False,
            "safe_mode_reason": None,
            "emotion_loop_paused": False,
            "writes_locked": False,
            "system_health": {"mirror_healthy": True, "anchor_healthy": True},
        }

        mock.return_value = conductor_instance
        yield conductor_instance


@pytest.fixture
def mock_memory_system():
    """Mock MemorySystem for testing."""
    with patch("core.memory_system.MemorySystem") as mock:
        memory_instance = MagicMock()

        # Mock memory layers
        memory_instance.long_term_memory = {"identity": [], "beliefs": [], "ephemeral": []}

        # Mock quota methods
        memory_instance.get_memory_quota_status.return_value = {
            "identity": {
                "current_items": 0,
                "max_items": 10,
                "usage_percentage": 0.0,
                "is_over_quota": False,
            },
            "beliefs": {
                "current_items": 0,
                "max_items": 50,
                "usage_percentage": 0.0,
                "is_over_quota": False,
            },
            "ephemeral": {
                "current_items": 0,
                "max_items": 100,
                "usage_percentage": 0.0,
                "is_over_quota": False,
            },
        }

        memory_instance.add_layered_memory.return_value = True
        memory_instance.prune_all_layers.return_value = {
            "identity": 0,
            "beliefs": 0,
            "ephemeral": 0,
        }

        mock.return_value = memory_instance
        yield memory_instance


@pytest.fixture
def mock_flask_app():
    """Mock Flask application for API testing."""
    from flask import Flask

    app = Flask(__name__)
    app.config["TESTING"] = True

    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def test_tokens():
    """Provide test authentication tokens."""
    return {
        "admin": "admin_test_token_12345",
        "system": "sys_test_token_67890",
        "user": "user_test_token_abcde",
        "invalid": "invalid_token_xyz",
    }


@pytest.fixture
def rate_limit_tracker():
    """Track rate limiting for testing."""
    tracker = {}

    def track_request(source: str, timestamp: float = None):
        """Track a request for rate limiting."""
        import time

        if timestamp is None:
            timestamp = time.time()

        if source not in tracker:
            tracker[source] = []
        tracker[source].append(timestamp)
        return tracker[source]

    def get_request_count(source: str, window: int = 60) -> int:
        """Get request count for source within time window."""
        import time

        if source not in tracker:
            return 0

        cutoff = time.time() - window
        recent_requests = [t for t in tracker[source] if t > cutoff]
        tracker[source] = recent_requests  # Clean up old requests
        return len(recent_requests)

    def reset_tracker():
        """Reset the rate limit tracker."""
        tracker.clear()

    return {
        "track": track_request,
        "count": get_request_count,
        "reset": reset_tracker,
        "data": tracker,
    }


@pytest.fixture
def chaos_controller(request):
    """Control chaos testing parameters."""
    chaos_level = request.config.getoption("--chaos-level")

    class ChaosController:
        def __init__(self, level: int):
            self.level = level
            self.failure_rate = min(level * 0.1, 0.9)  # 10% per level, max 90%
            self.random = random.Random(42)  # Deterministic chaos

        def should_fail(self) -> bool:
            """Determine if an operation should fail."""
            return self.random.random() < self.failure_rate

        def inject_delay(self) -> float:
            """Inject random delay for timing-based chaos."""
            return self.random.uniform(0.1, self.level * 0.5)

        def corrupt_data(self, data: Any) -> Any:
            """Randomly corrupt data."""
            if self.should_fail() and isinstance(data, dict):
                corrupted = data.copy()
                if corrupted:
                    key = self.random.choice(list(corrupted.keys()))
                    corrupted[key] = None
                return corrupted
            return data

    return ChaosController(chaos_level)


@pytest.fixture
def mirror_anchor_chaos():
    """Simulate Mirror/Anchor system failures."""

    class MirrorAnchorChaos:
        def __init__(self):
            self.mirror_failure_rate = 0.3
            self.anchor_failure_rate = 0.2
            self.random = random.Random(42)

        def simulate_mirror_health(self) -> bool:
            """Simulate mirror health check with potential failures."""
            return self.random.random() > self.mirror_failure_rate

        def simulate_anchor_health(self) -> bool:
            """Simulate anchor health check with potential failures."""
            return self.random.random() > self.anchor_failure_rate

        def inject_failure(self, system: str) -> bool:
            """Inject failure for specific system."""
            if system == "mirror":
                return not self.simulate_mirror_health()
            elif system == "anchor":
                return not self.simulate_anchor_health()
            return False

        def set_failure_rates(self, mirror_rate: float, anchor_rate: float):
            """Set custom failure rates."""
            self.mirror_failure_rate = mirror_rate
            self.anchor_failure_rate = anchor_rate

    return MirrorAnchorChaos()


@pytest.fixture
def drift_invariant_checker():
    """Check drift invariants during testing."""

    class DriftChecker:
        def __init__(self):
            self.drift_values = []

        def record_drift(self, value: float):
            """Record a drift value for invariant checking."""
            self.drift_values.append(value)

        def check_invariants(self) -> dict[str, bool]:
            """Check all drift invariants."""
            if not self.drift_values:
                return {"no_data": True}

            return {
                "in_bounds": all(0.0 <= v <= 1.0 for v in self.drift_values),
                "no_nans": all(not np.isnan(v) for v in self.drift_values),
                "no_infs": all(not np.isinf(v) for v in self.drift_values),
                "monotonic_sections": self._check_monotonic_sections(),
                "reasonable_variance": self._check_variance(),
            }

        def _check_monotonic_sections(self) -> bool:
            """Check that drift has reasonable monotonic sections."""
            if len(self.drift_values) < 3:
                return True

            # Look for sections of at least 3 consecutive values that are monotonic
            for i in range(len(self.drift_values) - 2):
                section = self.drift_values[i : i + 3]
                if all(a <= b for a, b in zip(section, section[1:])) or all(
                    a >= b for a, b in zip(section, section[1:])
                ):
                    return True
            return False

        def _check_variance(self) -> bool:
            """Check that drift variance is reasonable."""
            if len(self.drift_values) < 2:
                return True

            variance = np.var(self.drift_values)
            return 0.0 <= variance <= 0.25  # Reasonable variance bounds

    return DriftChecker()


@pytest.fixture
def cleanup_test_files():
    """Clean up test files after tests."""
    test_files = []
    test_dirs = []

    def register_file(filepath: str):
        """Register a file for cleanup."""
        test_files.append(Path(filepath))

    def register_dir(dirpath: str):
        """Register a directory for cleanup."""
        test_dirs.append(Path(dirpath))

    yield {"file": register_file, "dir": register_dir}

    # Cleanup
    for file_path in test_files:
        if file_path.exists():
            file_path.unlink()

    for dir_path in test_dirs:
        if dir_path.exists():
            import shutil

            shutil.rmtree(dir_path)


# Test markers
pytest.mark.deterministic = pytest.mark.skipif(
    not os.environ.get("PYTEST_DETERMINISTIC"), reason="Requires --deterministic flag"
)

pytest.mark.chaos = pytest.mark.skipif(
    os.environ.get("PYTEST_NO_CHAOS") == "true", reason="Chaos tests disabled"
)

pytest.mark.slow = pytest.mark.skipif(
    os.environ.get("PYTEST_FAST") == "true", reason="Slow tests skipped in fast mode"
)
