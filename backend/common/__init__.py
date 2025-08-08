"""
Common utilities for ProjectAlpha backend.

This package lazily exposes selected symbols from `.security` to avoid
importing heavy optional dependencies during static analysis and startup.
"""

import importlib
from typing import Any

__all__ = [
    "mask_token",
    "is_admin",
    "require_scope",
    "require_layer_access",
    "validate_json_schema",
    "audit_action",
    "create_request_context",
    "extract_token",
    "get_token_type",
    "can_access_layer",
    "log_audit_entry",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple re-export
    """Lazy attribute access to re-export names from `.security`.

    This keeps imports light and prevents mypy from following heavy imports
    when analyzing sibling modules in this package.
    """
    if name in __all__:
        # Use dynamic import to avoid static analysis pulling in heavy deps
        _security = importlib.import_module(__name__ + ".security")
        return getattr(_security, name)
    raise AttributeError(f"module 'backend.common' has no attribute {name!r}")
