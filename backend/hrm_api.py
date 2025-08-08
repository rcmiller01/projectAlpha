"""
HRM (Human Relationship Management) API with enhanced security.
Provides RBAC-protected endpoints for managing identity, beliefs, and ephemeral layers.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, Response, g, jsonify, request
from flask_cors import CORS

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import security utilities
from common.security import (
    audit_action,
    extract_token,
    get_token_type,
    mask_token,
    require_layer_access,
    require_scope,
    validate_json_schema,
)
from backend.data_portability import erase_user_data, export_user_data
from backend.anchor_system import anchor, AnchorResponse
from hrm.policy_dsl import PolicyEngine, evaluate_write

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

# Initialize PolicyEngine from env or default example policy
try:
    default_policy_path = project_root / "hrm" / "policies" / "example.yaml"
    policy_path_env = os.getenv("HRM_POLICY_PATH")
    policy_path = Path(policy_path_env) if policy_path_env else default_policy_path
    policy_engine = PolicyEngine.from_yaml(str(policy_path))
    logger.info(f"PolicyEngine loaded from {policy_path}")
except Exception as e:
    logger.warning(f"Failed to load policy file: {e}; falling back to default example policy")
    try:
        policy_engine = PolicyEngine.from_yaml(str(default_policy_path))
    except Exception as e2:
        logger.error(f"Failed to load default policy as well: {e2}; using empty ruleset")
        policy_engine = PolicyEngine([])

# JSON Schemas
IDENTITY_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "core_values": {"type": "object"},
        "identity_anchors": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "personality_traits": {"type": "object"},
        "fundamental_beliefs": {"type": "object"},
    },
}

BELIEFS_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "beliefs": {"type": "object"},
        "preferences": {"type": "object"},
        "learned_patterns": {"type": "object"},
    "associations": {"type": "array", "items": {"type": "object"}},
    "evidence": {"type": "object"},
    },
}

EPHEMERAL_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "current_mood": {"type": "string", "maxLength": 50},
        "temporary_state": {"type": "object"},
        "session_data": {"type": "object"},
        "context": {"type": "object"},
    },
}

# Data portability schemas
DATA_EXPORT_SCHEMA = {
    "type": "object",
    "required": ["user_id"],
    "properties": {
        "user_id": {"type": "string", "minLength": 1},
        "include_identity": {"type": "boolean"},
    },
}

DATA_ERASE_SCHEMA = {
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
        logger.error(f"Failed to load {layer_path}: {e}")

    return {"data": {}, "last_updated": None, "version": 1}


def save_layer_data(layer_path: Path, data: dict[str, Any]) -> None:
    """Save layer data to file."""
    try:
        data["last_updated"] = datetime.utcnow().isoformat()
        data["version"] = data.get("version", 1) + 1

        with open(layer_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save {layer_path}: {e}")
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
        # Policy enforcement and anchor confirmation
        sec = getattr(g, "security_context", {})
        token_type = sec.get("token_type")
        decision = evaluate_write(
            policy_engine,
            layer="identity",
            admin_token_ok=(token_type == "admin"),
            evidence=None,
        )
        if not decision.allowed:
            audit_action(
                "identity_policy_denied",
                success=False,
                policy_status=decision.status,
                reason=decision.reason,
            )
            return jsonify({"error": "Forbidden", "policy_status": decision.status}), 403

        anchor_action = {
            "action_type": "memory_write",
            "description": f"HRM identity write by {token_type or 'unknown'}",
        }
        anchor_resp = anchor.confirm(anchor_action, requester_id=token_type or "unknown")
        if anchor_resp != AnchorResponse.APPROVED:
            audit_action(
                "identity_anchor_denied", success=False, anchor_result=anchor_resp.value
            )
            return jsonify({"error": "Anchor denied", "anchor_result": anchor_resp.value}), 403

        current_data = load_layer_data(IDENTITY_PATH)
        update_data = g.validated_data

        # Merge update with current data
        current_data["data"].update(update_data)
        current_data["updated_by"] = getattr(g, "security_context", {}).get("request_id")

        save_layer_data(IDENTITY_PATH, current_data)

        audit_action(
            "identity_updated",
            updated_fields=list(update_data.keys()),
            success=True,
            policy_status=decision.status,
            anchor_result=anchor_resp.value,
        )

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
        sec = getattr(g, "security_context", {})
        token_type = sec.get("token_type")
        current_data = load_layer_data(BELIEFS_PATH)
        update_data = g.validated_data

        evidence = update_data.get("evidence") if isinstance(update_data, dict) else None
        update_core = {k: v for k, v in update_data.items() if k != "evidence"}

        decision = evaluate_write(
            policy_engine,
            layer="beliefs",
            admin_token_ok=(token_type == "admin"),
            evidence=evidence if isinstance(evidence, dict) else None,
        )
        if not decision.allowed:
            status_code = 400 if decision.status == "needs_evidence" else 403
            audit_action(
                "beliefs_policy_denied",
                success=False,
                policy_status=decision.status,
                reason=decision.reason,
            )
            return (
                jsonify({
                    "error": "Evidence required" if decision.status == "needs_evidence" else "Forbidden",
                    "policy_status": decision.status,
                    "reason": decision.reason,
                }),
                status_code,
            )

        anchor_action = {
            "action_type": "memory_write",
            "description": f"HRM beliefs write by {token_type or 'unknown'}",
        }
        anchor_resp = anchor.confirm(anchor_action, requester_id=token_type or "unknown")
        if anchor_resp != AnchorResponse.APPROVED:
            audit_action(
                "beliefs_anchor_denied", success=False, anchor_result=anchor_resp.value
            )
            return jsonify({"error": "Anchor denied", "anchor_result": anchor_resp.value}), 403

        # Merge update with current data
        current_data["data"].update(update_core)
        if evidence:
            current_data["last_evidence"] = evidence
        current_data["updated_by"] = getattr(g, "security_context", {}).get("request_id")

        save_layer_data(BELIEFS_PATH, current_data)

        audit_action(
            "beliefs_updated",
            updated_fields=list(update_core.keys()),
            success=True,
            policy_status=decision.status,
            anchor_result=anchor_resp.value,
        )

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
        sec = getattr(g, "security_context", {})
        token_type = sec.get("token_type")
        current_data = load_layer_data(EPHEMERAL_PATH)
        update_data = g.validated_data

        decision = evaluate_write(
            policy_engine,
            layer="ephemeral",
            admin_token_ok=(token_type == "admin"),
            evidence=None,
        )
        if not decision.allowed:
            audit_action(
                "ephemeral_policy_denied",
                success=False,
                policy_status=decision.status,
                reason=decision.reason,
            )
            return jsonify({"error": "Forbidden", "policy_status": decision.status}), 403

        anchor_action = {
            "action_type": "memory_write",
            "description": f"HRM ephemeral write by {token_type or 'unknown'}",
        }
        anchor_resp = anchor.confirm(anchor_action, requester_id=token_type or "unknown")
        if anchor_resp != AnchorResponse.APPROVED:
            audit_action(
                "ephemeral_anchor_denied", success=False, anchor_result=anchor_resp.value
            )
            return jsonify({"error": "Anchor denied", "anchor_result": anchor_resp.value}), 403

        # Merge update with current data
        current_data["data"].update(update_data)
        current_data["updated_by"] = getattr(g, "security_context", {}).get("request_id")

        # Attach expiry metadata if policy specifies TTL
        if decision.ttl_seconds:
            expires_at = (datetime.utcnow() + timedelta(seconds=int(decision.ttl_seconds))).isoformat()
            current_data["ttl_seconds"] = int(decision.ttl_seconds)
            current_data["expires_at"] = expires_at

        save_layer_data(EPHEMERAL_PATH, current_data)

        audit_action(
            "ephemeral_updated",
            updated_fields=list(update_data.keys()),
            success=True,
            policy_status=decision.status,
            ttl_seconds=decision.ttl_seconds,
            anchor_result=anchor_resp.value,
        )

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
            "timestamp": datetime.utcnow().isoformat(),
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
        if token_type == "admin":
            permitted = True
        elif token_type == "user" and ("user:basic" in scopes or "ephemeral:read" in scopes):
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
        if token_type == "admin":
            permitted = True
        elif token_type == "user" and ("ephemeral:write" in scopes or "user:basic" in scopes):
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
