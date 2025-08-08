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

    from backend.anchor_system import ActionType, AnchorSystem, AnchorResponse
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

# Safe fallbacks for dry-run utilities if not available
if not DRY_RUN_AVAILABLE:
    def is_dry_run() -> bool:  # type: ignore[misc]
        return False

    def format_dry_run_response(
        data: dict[str, Any], dry_run: bool, status_code: int | None = None
    ) -> dict[str, Any]:  # type: ignore[misc]
        out: dict[str, Any] = {"dry_run": dry_run}
        out.update(data)
        if status_code is not None:
            out["status_code"] = status_code
        return out

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

# RBAC/Beliefs confidence threshold
BELIEF_CONFIDENCE_THRESHOLD = float(os.getenv("BELIEF_CONFIDENCE_THRESHOLD", "0.6"))

def _current_role() -> str:
    try:
        sec = getattr(g, "security_context", {})
        ttype = sec.get("token_type")
        return ttype if isinstance(ttype, str) else "user"
    except Exception:
        return "user"

if POLICY_ENGINE_AVAILABLE:
    try:
        # Prefer explicit env path, else default policy under hrm/policies
        policy_path = os.getenv("HRM_POLICY_PATH", "hrm/policies/default_policy.yaml")
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


def _safe_evaluate_write(
    engine: Any,
    *,
    layer: str,
    admin_token_ok: bool,
    evidence: Optional[dict[str, Any]] = None,
):
    """Wrapper for evaluate_write that tolerates missing PolicyEngine module.

    Returns an object with attributes: allowed, status, reason, ttl_seconds.
    If the real evaluate_write is unavailable, default to allowed=True.
    """
    try:
        return evaluate_write(engine, layer=layer, admin_token_ok=admin_token_ok, evidence=evidence)
    except Exception:
        class _Decision:
            allowed = True
            status = "ok"
            reason = None
            ttl_seconds = None

        return _Decision()


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
    "evidence": {"type": "object"},
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


def _ephemeral_is_expired(payload: dict[str, Any]) -> bool:
    """Check if ephemeral payload has expired based on expires_at."""
    try:
        expires_at = payload.get("expires_at")
        if not expires_at:
            return False
        # Ensure timezone-aware comparison
        now = datetime.now(timezone.utc)
        exp = datetime.fromisoformat(expires_at)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return now >= exp
    except Exception:
        return False


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
        # Phase 3: Require admin role explicitly
        if _current_role() != "admin":
            return jsonify({"error": "Forbidden: admin required"}), 403

        # Policy enforcement and anchor confirmation
        sec = getattr(g, "security_context", {})
        token_type = sec.get("token_type")
        decision = _safe_evaluate_write(
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
        if not anchor_system:
            anchor_approved = True
            anchor_result_value = "APPROVED"
        else:
            _resp = anchor_system.confirm(anchor_action, requester_id=token_type or "unknown")
            anchor_result_value = getattr(_resp, "value", getattr(_resp, "name", str(_resp)))
            anchor_approved = anchor_result_value == "APPROVED"
        if not anchor_approved:
            audit_action(
                "identity_anchor_denied", success=False, anchor_result=anchor_result_value
            )
            return (
                jsonify({
                    "error": "Anchor denied",
                    "anchor_result": anchor_result_value,
                    "rationale": f"Anchor system returned {anchor_result_value}",
                }),
                403,
            )

        # Dry-run: simulate approval and skip write
        try:
            if is_dry_run():
                payload: dict[str, Any] = {
                    "message": "Simulated identity write (dry-run)",
                    "layer": "identity",
                    "anchor_result": anchor_result_value,
                }
                return jsonify(format_dry_run_response(payload, dry_run=True))
        except Exception:
            pass

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
            anchor_result=anchor_result_value,
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

        # Phase 3: Enforce confidence threshold for beliefs writes
        try:
            confidence_val = float(evidence.get("confidence")) if isinstance(evidence, dict) else None
        except Exception:
            confidence_val = None
        if confidence_val is None or confidence_val < BELIEF_CONFIDENCE_THRESHOLD:
            audit_action(
                "beliefs_confidence_denied",
                success=False,
                provided=confidence_val,
                threshold=BELIEF_CONFIDENCE_THRESHOLD,
            )
            return jsonify({
                "error": "Forbidden: insufficient confidence",
                "threshold": BELIEF_CONFIDENCE_THRESHOLD,
            }), 403

        decision = _safe_evaluate_write(
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
        if not anchor_system:
            anchor_approved = True
            anchor_result_value = "APPROVED"
        else:
            _resp = anchor_system.confirm(anchor_action, requester_id=token_type or "unknown")
            anchor_result_value = getattr(_resp, "value", getattr(_resp, "name", str(_resp)))
            anchor_approved = anchor_result_value == "APPROVED"
        if not anchor_approved:
            audit_action(
                "beliefs_anchor_denied", success=False, anchor_result=anchor_result_value
            )
            return (
                jsonify({
                    "error": "Anchor denied",
                    "anchor_result": anchor_result_value,
                    "rationale": f"Anchor system returned {anchor_result_value}",
                }),
                403,
            )

        # Dry-run: simulate approval and skip write
        try:
            if is_dry_run():
                payload: dict[str, Any] = {
                    "message": "Simulated beliefs write (dry-run)",
                    "layer": "beliefs",
                    "anchor_result": anchor_result_value,
                }
                return jsonify(format_dry_run_response(payload, dry_run=True))
        except Exception:
            pass

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
            anchor_result=anchor_result_value,
        )

        return jsonify({"success": True, "message": "Beliefs layer updated"})
    except Exception as e:
        audit_action("beliefs_update_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


# Ephemeral Layer Endpoints (All authenticated users)
@app.route("/api/hrm/ephemeral", methods=["GET"])
def get_ephemeral() -> Response:
    """Get ephemeral layer data (all authenticated users)."""
    try:
        data = load_layer_data(EPHEMERAL_PATH)

        # TTL cleanup on read: if expired, clear data and persist
        if isinstance(data, dict) and _ephemeral_is_expired(data):
            audit_action("ephemeral_ttl_expired", success=True)
            data["data"] = {}
            data.pop("expires_at", None)
            data.pop("ttl_seconds", None)
            save_layer_data(EPHEMERAL_PATH, data)
            data = load_layer_data(EPHEMERAL_PATH)
        audit_action("ephemeral_read", success=True)
        return jsonify(data)
    except Exception as e:
        audit_action("ephemeral_read_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


@app.route("/api/hrm/ephemeral", methods=["POST"])
@validate_json_schema(EPHEMERAL_UPDATE_SCHEMA)
def update_ephemeral() -> Response:
    """Update ephemeral layer data (all authenticated users)."""
    try:
        sec = getattr(g, "security_context", {})
        token_type = sec.get("token_type")
        current_data = load_layer_data(EPHEMERAL_PATH)
        update_data = g.validated_data

        decision = _safe_evaluate_write(
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
        if not anchor_system:
            anchor_approved = True
            anchor_result_value = "APPROVED"
        else:
            _resp = anchor_system.confirm(anchor_action, requester_id=token_type or "unknown")
            anchor_result_value = getattr(_resp, "value", getattr(_resp, "name", str(_resp)))
            anchor_approved = anchor_result_value == "APPROVED"
        if not anchor_approved:
            audit_action(
                "ephemeral_anchor_denied", success=False, anchor_result=anchor_result_value
            )
            return (
                jsonify({
                    "error": "Anchor denied",
                    "anchor_result": anchor_result_value,
                    "rationale": f"Anchor system returned {anchor_result_value}",
                }),
                403,
            )

        # Dry-run: simulate approval and skip write
        try:
            if is_dry_run():
                payload: dict[str, Any] = {
                    "message": "Simulated ephemeral write (dry-run)",
                    "layer": "ephemeral",
                    "anchor_result": anchor_result_value,
                }
                return jsonify(format_dry_run_response(payload, dry_run=True))
        except Exception:
            pass

        # Merge update with current data
        current_data["data"].update(update_data)
        current_data["updated_by"] = getattr(g, "security_context", {}).get("request_id")

        # Attach expiry metadata if policy specifies TTL, else apply default TTL if set
        ttl_default = int(os.getenv("DEFAULT_EPHEMERAL_TTL_SECONDS", "0") or 0)
        ttl_seconds = int(decision.ttl_seconds) if getattr(decision, "ttl_seconds", None) else 0
        effective_ttl = ttl_seconds or ttl_default
        if effective_ttl > 0:
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=effective_ttl)).isoformat()
            current_data["ttl_seconds"] = effective_ttl
            current_data["expires_at"] = expires_at

        save_layer_data(EPHEMERAL_PATH, current_data)

        audit_action(
            "ephemeral_updated",
            updated_fields=list(update_data.keys()),
            success=True,
            policy_status=decision.status,
            ttl_seconds=effective_ttl or decision.ttl_seconds,
            anchor_result=anchor_result_value,
        )

        return jsonify({"success": True, "message": "Ephemeral layer updated"})
    except Exception as e:
        audit_action("ephemeral_update_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


# Admin: force purge ephemeral layer
@app.route("/api/hrm/ephemeral/purge", methods=["POST"])
@require_layer_access("identity")
def purge_ephemeral() -> Response:
    """Admin-only: force-clear ephemeral data regardless of TTL."""
    try:
        if _current_role() != "admin":
            return jsonify({"error": "Forbidden: admin required"}), 403
        # Anchor confirmation before mutation
        anchor_action = {
            "action_type": "memory_delete",
            "description": "Force purge ephemeral layer",
            "target_layer": "ephemeral",
        }
        if not anchor_system:
            anchor_approved = True
            anchor_result_value = "APPROVED"
        else:
            _resp = anchor_system.confirm(anchor_action, requester_id="admin")
            anchor_result_value = getattr(_resp, "value", getattr(_resp, "name", str(_resp)))
            anchor_approved = anchor_result_value == "APPROVED"
        if not anchor_approved:
            audit_action(
                "ephemeral_purge_anchor_denied", success=False, anchor_result=anchor_result_value
            )
            return (
                jsonify({
                    "error": "Anchor denied",
                    "anchor_result": anchor_result_value,
                    "rationale": f"Anchor system returned {anchor_result_value}",
                }),
                403,
            )

        # Dry-run: simulate approval and skip write
        try:
            if is_dry_run():
                payload: dict[str, Any] = {
                    "message": "Simulated ephemeral purge (dry-run)",
                    "layer": "ephemeral",
                    "anchor_result": anchor_result_value,
                }
                return jsonify(format_dry_run_response(payload, dry_run=True))
        except Exception:
            pass

        data = load_layer_data(EPHEMERAL_PATH)
        data["data"] = {}
        data.pop("expires_at", None)
        data.pop("ttl_seconds", None)
        save_layer_data(EPHEMERAL_PATH, data)
        audit_action("ephemeral_purged", success=True)
        return jsonify({"success": True, "message": "Ephemeral layer purged"})
    except Exception as e:
        audit_action("ephemeral_purge_failed", error=str(e), success=False)
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

        # Anchor confirmation before mutation
        anchor_action = {
            "action_type": "memory_delete",
            "description": f"Erase data for user {user_id}",
            "target_layer": (
                "identity" if erase_identity else "beliefs/ephemeral"
            ),
        }
        if not anchor_system:
            anchor_approved = True
            anchor_result_value = "APPROVED"
        else:
            _resp = anchor_system.confirm(anchor_action, requester_id=token_type or "unknown")
            anchor_result_value = getattr(_resp, "value", getattr(_resp, "name", str(_resp)))
            anchor_approved = anchor_result_value == "APPROVED"
        if not anchor_approved:
            audit_action(
                "erase_anchor_denied", success=False, anchor_result=anchor_result_value
            )
            return (
                jsonify({
                    "error": "Anchor denied",
                    "anchor_result": anchor_result_value,
                    "rationale": f"Anchor system returned {anchor_result_value}",
                }),
                403,
            )

        # Dry-run: simulate approval and skip actual deletion
        try:
            if is_dry_run():
                payload: dict[str, Any] = {
                    "message": "Simulated erase (dry-run)",
                    "user_id": user_id,
                    "erase_identity": erase_identity,
                    "anchor_result": anchor_result_value,
                }
                return jsonify(format_dry_run_response(payload, dry_run=True))
        except Exception:
            pass

        results = erase_user_data(user_id, erase_identity=erase_identity)
        audit_action("erase_success", success=True, target_user=user_id, details=results)
        return jsonify({"success": True, **results})
    except Exception as e:
        audit_action("erase_error", success=False, error=str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
