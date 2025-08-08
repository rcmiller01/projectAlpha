"""
Central Core Config Loader

Loads required keys with precedence: environment variables > .env file > fallback.
Type-checks values and raises on invalid or missing (when no fallback).

Required keys:
- HRM_POLICY_PATH: str
- MUSE_ENABLED: bool
- MUSE_QUALITY_MODE: str
- DRY_RUN: bool
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, root_validator, validator


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


class CoreConfig(BaseModel):
    HRM_POLICY_PATH: str = Field(..., description="Path to HRM policy YAML file")
    MUSE_ENABLED: bool = Field(..., description="Enable Muse model features")
    MUSE_QUALITY_MODE: str = Field(..., description="Muse quality mode string")
    DRY_RUN: bool = Field(..., description="Enable dry-run simulation mode")

    @validator("HRM_POLICY_PATH")
    def _validate_policy_path(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("HRM_POLICY_PATH must be a non-empty string")
        return v

    @validator("MUSE_QUALITY_MODE")
    def _validate_muse_quality_mode(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("MUSE_QUALITY_MODE must be a non-empty string")
        return v

    @root_validator(pre=True)
    def _coerce_types(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "MUSE_ENABLED" in values and isinstance(values["MUSE_ENABLED"], str):
            values["MUSE_ENABLED"] = _truthy(values["MUSE_ENABLED"])  # type: ignore[index]
        if "DRY_RUN" in values and isinstance(values["DRY_RUN"], str):
            values["DRY_RUN"] = _truthy(values["DRY_RUN"])  # type: ignore[index]
        return values


_FALLBACKS: dict[str, Any] = {
    "HRM_POLICY_PATH": "hrm/policies/default_policy.yaml",
    "MUSE_ENABLED": False,
    "MUSE_QUALITY_MODE": "standard",
    "DRY_RUN": False,
}

_CONFIG_SINGLETON: Optional[CoreConfig] = None


def load_core_config(*, force_reload: bool = False) -> CoreConfig:
    """Load core config with env > .env > fallback precedence.

    Raises ValueError if any required key remains unset and no fallback exists.
    """
    global _CONFIG_SINGLETON
    if _CONFIG_SINGLETON is not None and not force_reload:
        return _CONFIG_SINGLETON

    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)

    def _get_str(name: str) -> Optional[str]:
        v = os.getenv(name)
        return v if v is not None else None

    raw: dict[str, Any] = {}
    for key in ("HRM_POLICY_PATH", "MUSE_ENABLED", "MUSE_QUALITY_MODE", "DRY_RUN"):
        val = _get_str(key)
        if val is None:
            if key in _FALLBACKS:
                raw[key] = _FALLBACKS[key]
            else:
                raise ValueError(f"Missing required configuration: {key}")
        else:
            raw[key] = val

    cfg = CoreConfig(**raw)
    _CONFIG_SINGLETON = cfg
    return cfg


def get_core_config() -> CoreConfig:
    return load_core_config()


__all__ = ["CoreConfig", "load_core_config", "get_core_config"]
