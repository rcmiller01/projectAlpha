"""
HRM (Human Relationship Management) API with enhanced security.
Provides RBAC-protected endpoints for managing identity, beliefs, and ephemeral layers.
"""

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from flask import Flask, g, jsonify, request
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
def get_identity():
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
def update_identity():
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
def get_beliefs():
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
def update_beliefs():
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
def get_ephemeral():
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
def update_ephemeral():
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
def get_system_status():
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


if __name__ == "__main__":
    app.run(debug=True, port=5001)
