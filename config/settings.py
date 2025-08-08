"""
ProjectAlpha Configuration System
Centralized config loading with strict validation and feature flags.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ProjectAlphaSettings(BaseModel):
    """
    Centralized configuration for ProjectAlpha with strict validation.

    Environment variable precedence: ENV > .env > defaults
    """

    # === Database Configuration ===
    MONGO_ROOT_USERNAME: str = Field(..., description="MongoDB root username")
    MONGO_ROOT_PASSWORD: str = Field(..., description="MongoDB root password")
    MONGO_DATABASE: str = Field(default="emotional_ai", description="MongoDB database name")

    # === Server Configuration ===
    SERVER_ROLE: str = Field(..., description="Role of the server")
    SERVER_ID: str = Field(..., description="Unique identifier for the server")
    PORT: int = Field(default=8000, description="Port number for the server", ge=1, le=65535)
    CLUSTER_ENABLED: bool = Field(default=False, description="Whether clustering is enabled")
    GPU_ENABLED: bool = Field(default=False, description="Whether GPU is enabled")

    # === Emotional Processing Configuration ===
    DRIFT_SCALING_FACTOR: float = Field(
        default=0.35, description="Scaling factor for emotional drift calculations", ge=0.0, le=1.0
    )
    MAX_PENALTY_THRESHOLD: float = Field(
        default=0.85,
        description="Maximum penalty threshold for affective delta calculations",
        ge=0.0,
        le=1.0,
    )

    # === Rate Limiting Configuration ===
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Rate limiting window in seconds", ge=1)
    RATE_LIMIT_MAX: int = Field(
        default=120, description="Maximum requests per rate limit window", ge=1
    )

    # === Feature Flags ===
    EMOTION_LOOP_ENABLED: bool = Field(default=True, description="Enable emotional processing loop")
    AUTOPILOT_ENABLED: bool = Field(default=True, description="Enable autopilot mode")
    SAFE_MODE_FORCE: bool = Field(default=False, description="Force safe mode (kill switch)")

    # === Security Configuration ===
    ADMIN_MASTER_KEY: Optional[str] = Field(
        default=None, description="Master admin key for sensitive operations"
    )

    # === Graceful Degradation Configuration ===
    SAFE_MODE_FORCE: bool = Field(default=False, description="Force safe mode (kill switch)")

    # Retry configuration
    RETRY_MAX_ATTEMPTS: int = Field(
        default=3, description="Maximum retry attempts for external calls", ge=1, le=10
    )
    RETRY_BASE_DELAY: float = Field(
        default=1.0, description="Base delay for exponential backoff (seconds)", ge=0.1, le=10.0
    )
    RETRY_MAX_DELAY: float = Field(
        default=60.0, description="Maximum retry delay (seconds)", ge=1.0, le=300.0
    )

    # Idempotency configuration
    IDEMPOTENCY_CACHE_TTL: int = Field(
        default=3600, description="Idempotency cache TTL (seconds)", ge=300, le=86400
    )

    # Memory quota configuration
    MEMORY_QUOTA_IDENTITY: int = Field(
        default=100, description="Maximum items in identity memory layer", ge=10, le=1000
    )
    MEMORY_QUOTA_BELIEFS: int = Field(
        default=500, description="Maximum items in beliefs memory layer", ge=50, le=5000
    )
    MEMORY_QUOTA_EPHEMERAL: int = Field(
        default=1000, description="Maximum items in ephemeral memory layer", ge=100, le=10000
    )

    # Health check configuration
    HEALTH_CHECK_TIMEOUT: float = Field(
        default=5.0, description="Health check timeout (seconds)", ge=1.0, le=30.0
    )
    MIRROR_HEALTH_CHECK_ENABLED: bool = Field(
        default=True, description="Enable mirror system health checks"
    )
    ANCHOR_HEALTH_CHECK_ENABLED: bool = Field(
        default=True, description="Enable anchor system health checks"
    )

    # === Debugging and Development ===
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    @validator("SERVER_ROLE")
    def validate_server_role(cls, v):
        valid_roles = ["primary", "secondary"]
        if v not in valid_roles:
            raise ValueError(f"SERVER_ROLE must be one of {valid_roles}")
        return v

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @validator("DRIFT_SCALING_FACTOR")
    def validate_drift_scaling(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("DRIFT_SCALING_FACTOR must be between 0.0 and 1.0")
        return v

    @validator("MAX_PENALTY_THRESHOLD")
    def validate_penalty_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("MAX_PENALTY_THRESHOLD must be between 0.0 and 1.0")
        return v


def load_settings() -> ProjectAlphaSettings:
    """
    Load and validate ProjectAlpha settings with proper precedence.

    Returns:
        ProjectAlphaSettings: Validated configuration object

    Raises:
        ValidationError: If configuration validation fails
    """
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        logger.debug(f"Loaded environment from {env_file}")

    # Get values from environment with defaults
    config_data = {
        # Required fields - must be set
        "MONGO_ROOT_USERNAME": os.getenv("MONGO_ROOT_USERNAME") or os.getenv("MONGODB_USERNAME"),
        "MONGO_ROOT_PASSWORD": os.getenv("MONGO_ROOT_PASSWORD") or os.getenv("MONGODB_PASSWORD"),
        "SERVER_ROLE": os.getenv("SERVER_ROLE", "primary"),
        "SERVER_ID": os.getenv("SERVER_ID", "default-server"),
        # Optional fields with defaults
        "MONGO_DATABASE": os.getenv("MONGO_DATABASE", "emotional_ai"),
        "PORT": int(os.getenv("PORT", "8000")),
        "CLUSTER_ENABLED": os.getenv("CLUSTER_ENABLED", "false").lower() == "true",
        "GPU_ENABLED": os.getenv("GPU_ENABLED", "false").lower() == "true",
        # Emotional processing
        "DRIFT_SCALING_FACTOR": float(os.getenv("DRIFT_SCALING_FACTOR", "0.35")),
        "MAX_PENALTY_THRESHOLD": float(os.getenv("MAX_PENALTY_THRESHOLD", "0.85")),
        # Rate limiting
        "RATE_LIMIT_WINDOW": int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        "RATE_LIMIT_MAX": int(os.getenv("RATE_LIMIT_MAX", "120")),
        # Feature flags
        "EMOTION_LOOP_ENABLED": os.getenv("EMOTION_LOOP_ENABLED", "true").lower() == "true",
        "AUTOPILOT_ENABLED": os.getenv("AUTOPILOT_ENABLED", "true").lower() == "true",
        "SAFE_MODE_FORCE": os.getenv("SAFE_MODE_FORCE", "false").lower() == "true",
        # Security
        "ADMIN_MASTER_KEY": os.getenv("ADMIN_MASTER_KEY"),
        # Graceful degradation
        "RETRY_MAX_ATTEMPTS": int(os.getenv("RETRY_MAX_ATTEMPTS", "3")),
        "RETRY_BASE_DELAY": float(os.getenv("RETRY_BASE_DELAY", "1.0")),
        "RETRY_MAX_DELAY": float(os.getenv("RETRY_MAX_DELAY", "60.0")),
        "IDEMPOTENCY_CACHE_TTL": int(os.getenv("IDEMPOTENCY_CACHE_TTL", "3600")),
        "MEMORY_QUOTA_IDENTITY": int(os.getenv("MEMORY_QUOTA_IDENTITY", "100")),
        "MEMORY_QUOTA_BELIEFS": int(os.getenv("MEMORY_QUOTA_BELIEFS", "500")),
        "MEMORY_QUOTA_EPHEMERAL": int(os.getenv("MEMORY_QUOTA_EPHEMERAL", "1000")),
        "HEALTH_CHECK_TIMEOUT": float(os.getenv("HEALTH_CHECK_TIMEOUT", "5.0")),
        "MIRROR_HEALTH_CHECK_ENABLED": os.getenv("MIRROR_HEALTH_CHECK_ENABLED", "true").lower()
        == "true",
        "ANCHOR_HEALTH_CHECK_ENABLED": os.getenv("ANCHOR_HEALTH_CHECK_ENABLED", "true").lower()
        == "true",
        # Debug
        "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }

    # Validate required fields
    required_fields = ["MONGO_ROOT_USERNAME", "MONGO_ROOT_PASSWORD"]
    for field in required_fields:
        if not config_data[field]:
            raise ValueError(f"Required environment variable {field} is not set")

    try:
        settings = ProjectAlphaSettings(**config_data)
        logger.info("Configuration loaded and validated successfully")
        return settings
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def log_configuration_summary(settings: ProjectAlphaSettings) -> None:
    """
    Log a summary of the effective configuration (masking secrets).

    Args:
        settings: The validated settings object
    """
    # Create a masked version of the config for logging
    config_summary = {
        "server": {
            "role": settings.SERVER_ROLE,
            "id": settings.SERVER_ID,
            "port": settings.PORT,
            "cluster_enabled": settings.CLUSTER_ENABLED,
            "gpu_enabled": settings.GPU_ENABLED,
        },
        "database": {
            "database": settings.MONGO_DATABASE,
            "username": "***masked***" if settings.MONGO_ROOT_USERNAME else None,
            "password": "***masked***" if settings.MONGO_ROOT_PASSWORD else None,
        },
        "emotional_processing": {
            "drift_scaling_factor": settings.DRIFT_SCALING_FACTOR,
            "max_penalty_threshold": settings.MAX_PENALTY_THRESHOLD,
        },
        "rate_limiting": {
            "window_seconds": settings.RATE_LIMIT_WINDOW,
            "max_requests": settings.RATE_LIMIT_MAX,
        },
        "features": {
            "emotion_loop": settings.EMOTION_LOOP_ENABLED,
            "autopilot": settings.AUTOPILOT_ENABLED,
            "safe_mode_forced": settings.SAFE_MODE_FORCE,
        },
        "security": {
            "admin_key": "***configured***" if settings.ADMIN_MASTER_KEY else "not_set",
        },
        "debug": {
            "debug_mode": settings.DEBUG,
            "log_level": settings.LOG_LEVEL,
        },
    }

    logger.info(f"ProjectAlpha Config: {config_summary}")


def get_settings() -> ProjectAlphaSettings:
    """
    Get the current settings instance.

    Returns:
        ProjectAlphaSettings: The configuration object
    """
    return load_settings()


# Feature flag helpers
def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled."""
    current_settings = get_settings()
    feature_map = {
        "emotion_loop": current_settings.EMOTION_LOOP_ENABLED,
        "autopilot": current_settings.AUTOPILOT_ENABLED,
        "safe_mode": current_settings.SAFE_MODE_FORCE,
        "debug": current_settings.DEBUG,
        "gpu": current_settings.GPU_ENABLED,
        "cluster": current_settings.CLUSTER_ENABLED,
    }
    return feature_map.get(feature_name, False)


def is_safe_mode_active() -> bool:
    """Check if safe mode (kill switch) is active."""
    current_settings = get_settings()
    return current_settings.SAFE_MODE_FORCE


def get_rate_limit_config() -> dict[str, int]:
    """Get rate limiting configuration."""
    current_settings = get_settings()
    return {
        "window": current_settings.RATE_LIMIT_WINDOW,
        "max_requests": current_settings.RATE_LIMIT_MAX,
    }


# Global settings instance for convenience
try:
    settings = get_settings()
    log_configuration_summary(settings)
except Exception as e:
    # Handle case where required env vars are not set during import
    print(f"Warning: Could not load settings on import: {e}")
    settings = None

if __name__ == "__main__":
    # Test configuration loading
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        test_settings = load_settings()
        print("✅ Configuration loaded successfully")
        log_configuration_summary(test_settings)
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
