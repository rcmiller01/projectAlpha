"""
HRM (Human Relationship Management) API with enhanced security.
Provides RBAC-protected endpoints for managing identity, beliefs, and ephemeral layers.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from flask import Flask, Response, g, jsonify, request
from flask_cors import CORS

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import security utilities
from backend.data_portability import erase_user_data, export_user_data
from common.security import (
    audit_action,
    extract_token,
    get_token_type,
    mask_token,
    require_layer_access,
    require_scope,
    validate_json_schema,
)

# Import Policy DSL and Anchor System
try:
    # Import with sys.path modification for dry-run utilities
    import sys
    from pathlib import Path

    from backend.anchor_system import ActionType, AnchorSystem
    from hrm.policy_dsl import PolicyEngine, evaluate_read, evaluate_write

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from common.dryrun import dry_guard, format_dry_run_response, is_dry_run

    POLICY_ENGINE_AVAILABLE = True
    DRY_RUN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Policy DSL, Anchor System, or dry-run utilities not available: {e}")
    POLICY_ENGINE_AVAILABLE = False
    DRY_RUN_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Data storage paths
IDENTITY_PATH = Path("data/identity_layer.json")
BELIEFS_PATH = Path("data/beliefs_layer.json")
EPHEMERAL_PATH = Path("data/ephemeral_layer.json")

# Ensure data directory exists
IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize Policy Engine and Anchor System
policy_engine = None
anchor_system = None

if POLICY_ENGINE_AVAILABLE:
    try:
        policy_path = os.getenv("HRM_POLICY_PATH", "hrm/policy_config.yaml")
        if Path(policy_path).exists():
            policy_engine = PolicyEngine.from_yaml(policy_path)
            logger.info(f"Policy Engine loaded from {policy_path}")
        else:
            logger.warning(f"Policy file not found at {policy_path}")

        anchor_system = AnchorSystem()
        logger.info("Anchor System initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Policy Engine or Anchor System: {e}")
        POLICY_ENGINE_AVAILABLE = False


def require_admin_token() -> bool:
    """Check if current request has admin token."""
    try:
        token = extract_token()
        if token is None:
            return False
        return get_token_type(token) == "admin"
    except Exception:
        return False


def require_anchor_confirmation(
    layer: str, payload: dict[str, Any], operation: str = "write"
) -> tuple[bool, str]:
    """Check anchor confirmation for HRM operations."""
    if not anchor_system:
        logger.warning("[HRM_API] Anchor system not available, allowing operation")
        return True, "Anchor system disabled"

    try:
        # Create action data for anchor system
        action_data: dict[str, Any] = {
            "action_type": "memory_write" if operation == "write" else "memory_delete",
            "description": f"HRM {operation} to {layer} layer",
            "target_layer": layer,
            "payload_size": len(str(payload)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Use the confirm method from AnchorSystem
        from backend.anchor_system import AnchorResponse

        response = anchor_system.confirm(action_data, requester_id="hrm_api")

        if response == AnchorResponse.APPROVED:
            return True, "Anchor approved"
        elif response == AnchorResponse.PENDING:
            return False, "Anchor confirmation pending"
        elif response == AnchorResponse.RATE_LIMITED:
            return False, "Anchor rate limited"
        else:
            return False, f"Anchor denied: {response.value}"

    except Exception as e:
        logger.error(f"[HRM_API] Anchor confirmation error: {e}")
        return False, f"Anchor error: {e!s}"


def log_policy_decision(
    actor: str,
    layer: str,
    operation: str,
    policy_decision: dict[str, Any],
    anchor_result: tuple[bool, str],
) -> None:
    """Log policy and anchor decisions for audit."""
    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "layer": layer,
        "operation": operation,
        "policy_decision": policy_decision,
        "anchor_confirmed": anchor_result[0],
        "anchor_reason": anchor_result[1],
    }
    logger.info(f"[HRM_API] Policy Decision - {json.dumps(log_entry)}")

    # Also use existing audit system
    audit_action(
        f"hrm_policy_{layer}_{operation}",
        details=log_entry,
        success=policy_decision.get("allowed", False) and anchor_result[0],
    )


# JSON Schemas
IDENTITY_UPDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "core_values": {"type": "object"},
        "identity_anchors": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "personality_traits": {"type": "object"},
        "fundamental_beliefs": {"type": "object"},
    },
}

BELIEFS_UPDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "beliefs": {"type": "object"},
        "preferences": {"type": "object"},
        "learned_patterns": {"type": "object"},
        "associations": {"type": "array", "items": {"type": "object"}},
    },
}

EPHEMERAL_UPDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "current_mood": {"type": "string", "maxLength": 50},
        "temporary_state": {"type": "object"},
        "session_data": {"type": "object"},
        "context": {"type": "object"},
    },
}

# Data portability schemas
DATA_EXPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["user_id"],
    "properties": {
        "user_id": {"type": "string", "minLength": 1},
        "include_identity": {"type": "boolean"},
    },
}

DATA_ERASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["user_id"],
    "properties": {
        "user_id": {"type": "string", "minLength": 1},
        "erase_identity": {"type": "boolean"},
    },
}


def load_layer_data(layer_path: Path) -> dict[str, Any]:
    """Load layer data from file."""
    try:
        if layer_path.exists():
            with open(layer_path) as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"[HRM_API] Failed to load {layer_path}: {e}")
        return {}

    return {"data": {}, "last_updated": None, "version": 1}


def save_layer_data(layer_path: Path, data: dict[str, Any]) -> None:
    """Save layer data to file."""
    try:
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        data["version"] = data.get("version", 1) + 1

        with open(layer_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"[HRM_API] Failed to save {layer_path}: {e}")
        raise


# Identity Layer Endpoints (Admin only)
@app.route("/api/hrm/identity", methods=["GET"])
@require_layer_access("identity")
def get_identity() -> Response:
    """Get identity layer data (admin only)."""
    try:
        data = load_layer_data(IDENTITY_PATH)
        audit_action("identity_read", success=True)
        return jsonify(data)
    except Exception as e:
        audit_action("identity_read_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


@app.route("/api/hrm/identity", methods=["POST"])
@require_layer_access("identity")
@validate_json_schema(IDENTITY_UPDATE_SCHEMA)
def update_identity() -> Response:
    """Update identity layer data (admin only)."""
    try:
        current_data = load_layer_data(IDENTITY_PATH)
        update_data = g.validated_data

        # Merge update with current data
        current_data["data"].update(update_data)
        current_data["updated_by"] = getattr(g, "security_context", {}).get("request_id")

        save_layer_data(IDENTITY_PATH, current_data)

        audit_action("identity_updated", updated_fields=list(update_data.keys()), success=True)

        return jsonify({"success": True, "message": "Identity layer updated"})
    except Exception as e:
        audit_action("identity_update_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


# Beliefs Layer Endpoints (Admin/System only)
@app.route("/api/hrm/beliefs", methods=["GET"])
@require_scope(["beliefs:read"])
def get_beliefs() -> Response:
    """Get beliefs layer data (admin/system only)."""
    try:
        data = load_layer_data(BELIEFS_PATH)
        audit_action("beliefs_read", success=True)
        return jsonify(data)
    except Exception as e:
        audit_action("beliefs_read_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


@app.route("/api/hrm/beliefs", methods=["POST"])
@require_scope(["beliefs:write"])
@validate_json_schema(BELIEFS_UPDATE_SCHEMA)
def update_beliefs() -> Response:
    """Update beliefs layer data (admin/system only)."""
    try:
        current_data = load_layer_data(BELIEFS_PATH)
        update_data = g.validated_data

        # Merge update with current data
        current_data["data"].update(update_data)
        current_data["updated_by"] = getattr(g, "security_context", {}).get("request_id")

        save_layer_data(BELIEFS_PATH, current_data)

        audit_action("beliefs_updated", updated_fields=list(update_data.keys()), success=True)

        return jsonify({"success": True, "message": "Beliefs layer updated"})
    except Exception as e:
        audit_action("beliefs_update_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


# Ephemeral Layer Endpoints (All authenticated users)
@app.route("/api/hrm/ephemeral", methods=["GET"])
@require_scope(["ephemeral:read"])
def get_ephemeral() -> Response:
    """Get ephemeral layer data (all authenticated users)."""
    try:
        data = load_layer_data(EPHEMERAL_PATH)
        audit_action("ephemeral_read", success=True)
        return jsonify(data)
    except Exception as e:
        audit_action("ephemeral_read_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


@app.route("/api/hrm/ephemeral", methods=["POST"])
@require_scope(["ephemeral:write"])
@validate_json_schema(EPHEMERAL_UPDATE_SCHEMA)
def update_ephemeral() -> Response:
    """Update ephemeral layer data (all authenticated users)."""
    try:
        current_data = load_layer_data(EPHEMERAL_PATH)
        update_data = g.validated_data

        # Merge update with current data
        current_data["data"].update(update_data)
        current_data["updated_by"] = getattr(g, "security_context", {}).get("request_id")

        save_layer_data(EPHEMERAL_PATH, current_data)

        audit_action("ephemeral_updated", updated_fields=list(update_data.keys()), success=True)

        return jsonify({"success": True, "message": "Ephemeral layer updated"})
    except Exception as e:
        audit_action("ephemeral_update_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


# System Status Endpoint
@app.route("/api/hrm/status", methods=["GET"])
@require_scope(["system:operate", "user:basic"])
def get_system_status() -> Response:
    """Get HRM system status."""
    try:
        token = extract_token()
        token_type = get_token_type(token) if token else None

        status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "layers": {
                "identity": {"accessible": token_type == "admin"},
                "beliefs": {"accessible": token_type in ["admin", "system"]},
                "ephemeral": {"accessible": token_type in ["admin", "system", "user"]},
            },
            "permissions": list(getattr(g, "security_context", {}).get("scopes", set())),
        }

        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Data Portability Endpoints
@app.route("/api/data/export", methods=["POST"])
@validate_json_schema(DATA_EXPORT_SCHEMA)
def export_data() -> Response:
    """Export user data across layers (admin or user with scope)."""
    try:
        token = extract_token()
        token_type = get_token_type(token) if token else None
        scopes = getattr(g, "security_context", {}).get("scopes", set())

        body = g.validated_data
        user_id = body["user_id"]
        include_identity = bool(body.get("include_identity", False))

        # Authorization: admin OR user exporting own data with appropriate scope
        permitted = False
        if (
            token_type == "admin"
            or token_type == "user"
            and ("user:basic" in scopes or "ephemeral:read" in scopes)
        ):
            permitted = True

        if not permitted:
            audit_action("export_denied", success=False, target_user=user_id)
            return jsonify({"error": "Forbidden"}), 403

        # Policy: by default, identity is excluded unless admin
        if include_identity and token_type != "admin":
            include_identity = False

        data = export_user_data(user_id, include_identity=include_identity)
        audit_action("export_success", success=True, target_user=user_id)
        return jsonify(data)
    except Exception as e:
        audit_action("export_error", success=False, error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/erase", methods=["POST"])
@validate_json_schema(DATA_ERASE_SCHEMA)
def erase_data() -> Response:
    """Erase user data (admin or user with erase scope for own data)."""
    try:
        token = extract_token()
        token_type = get_token_type(token) if token else None
        scopes = getattr(g, "security_context", {}).get("scopes", set())

        body = g.validated_data
        user_id = body["user_id"]
        erase_identity = bool(body.get("erase_identity", False))

        # Authorization: admin can erase; user can erase own ephemeral/beliefs with scope
        permitted = False
        if (
            token_type == "admin"
            or token_type == "user"
            and ("ephemeral:write" in scopes or "user:basic" in scopes)
        ):
            permitted = True

        if not permitted:
            audit_action("erase_denied", success=False, target_user=user_id)
            return jsonify({"error": "Forbidden"}), 403

        # Policy: identity erase requires admin
        if erase_identity and token_type != "admin":
            erase_identity = False

        results = erase_user_data(user_id, erase_identity=erase_identity)
        audit_action("erase_success", success=True, target_user=user_id, details=results)
        return jsonify({"success": True, **results})
    except Exception as e:
        audit_action("erase_error", success=False, error=str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
