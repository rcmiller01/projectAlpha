"""
Data Portability utilities.
Provides export and erase operations with RBAC and audit logging.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Response, jsonify

from backend.common.security import audit_action

logger = logging.getLogger(__name__)

IDENTITY_PATH = Path("data/identity_layer.json")
BELIEFS_PATH = Path("data/beliefs_layer.json")
EPHEMERAL_PATH = Path("data/ephemeral_layer.json")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return {}


def export_user_data(user_id: str, include_identity: bool = False) -> dict[str, Any]:
    """Return user's data across layers. Identity can be excluded per policy."""
    data: dict[str, Any] = {"user_id": user_id, "layers": {}}
    try:
        if include_identity:
            identity = _load_json(IDENTITY_PATH)
            # Filter identity data by user when present
            user_identity = identity.get("data", {}).get(user_id)
            if user_identity is not None:
                data["layers"]["identity"] = user_identity

        beliefs = _load_json(BELIEFS_PATH)
        eph = _load_json(EPHEMERAL_PATH)

        # Simple user scoping by metadata.user_id when present
        def filter_by_user(obj: dict[str, Any]) -> Any:
            content = obj.get("data", obj)
            if isinstance(content, dict) and user_id in content:
                return content.get(user_id)
            return content

        data["layers"]["beliefs"] = filter_by_user(beliefs)
        data["layers"]["ephemeral"] = filter_by_user(eph)

        audit_action("data_export", success=True, target_user=user_id)
        return data
    except Exception as e:
        audit_action("data_export_failed", success=False, target_user=user_id, error=str(e))
        raise


def erase_user_data(user_id: str, erase_identity: bool = False) -> dict[str, Any]:
    """Purge user-specific data in beliefs/ephemeral layers; identity optional by policy."""
    results: dict[str, Any] = {"user_id": user_id, "erased": []}

    def erase_in_file(path: Path, layer: str) -> int:
        obj = _load_json(path)
        count_before = 0
        count_after = 0
        try:
            # Common structures: {"data": {...}} or plain dict/list
            if isinstance(obj, dict):
                data = obj.get("data")
                if isinstance(data, dict) and user_id in data:
                    count_before = 1
                    del data[user_id]
                    obj["data"] = data
                elif user_id in obj:
                    count_before = 1
                    del obj[user_id]
            elif isinstance(obj, list):
                count_before = len([x for x in obj if getattr(x, "user_id", None) == user_id])
                obj = [x for x in obj if getattr(x, "user_id", None) != user_id]

            # Save back
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)

            # Count after simplistic
            count_after = 0
        except Exception as e:
            logger.error(f"Failed to erase in {path}: {e}")
        return count_before - count_after

    try:
        erased_beliefs = erase_in_file(BELIEFS_PATH, "beliefs")
        if erased_beliefs:
            results["erased"].append({"layer": "beliefs", "count": erased_beliefs})

        erased_eph = erase_in_file(EPHEMERAL_PATH, "ephemeral")
        if erased_eph:
            results["erased"].append({"layer": "ephemeral", "count": erased_eph})

        if erase_identity:
            erased_id = erase_in_file(IDENTITY_PATH, "identity")
            if erased_id:
                results["erased"].append({"layer": "identity", "count": erased_id})

        audit_action("data_erase", success=True, target_user=user_id, details=results)
        return results
    except Exception as e:
        audit_action("data_erase_failed", success=False, target_user=user_id, error=str(e))
        raise
