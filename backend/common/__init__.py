"""
Common utilities for ProjectAlpha backend.

This package lazily exposes selected symbols from `.security` to avoid
importing heavy optional dependencies during static analysis and startup.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Imported for type checkers only; avoids runtime import side-effects.
    from .security import (
        audit_action,
        can_access_layer,
        create_request_context,
        extract_token,
        get_token_type,
        is_admin,
        log_audit_entry,
        mask_token,
        require_layer_access,
        require_scope,
        validate_json_schema,
    )

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
        from . import security as _security

        return getattr(_security, name)
    raise AttributeError(f"module 'backend.common' has no attribute {name!r}")
