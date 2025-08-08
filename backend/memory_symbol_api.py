#!/usr/bin/env python3
"""
Memory and Symbol API endpoints for MemoryAndSymbolViewer component.
Extends the existing CoreArbiter API with memory and symbolic tracking.
Enhanced with RBAC, JSON schema validation, and comprehensive audit logging.
"""

import json
import logging
import random
import re
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List

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

# Rate limiting configuration (now using centralized config)
try:
    from config.settings import get_settings

    settings = get_settings()
    RATE_LIMIT_REQUESTS = settings.RATE_LIMIT_MAX
    RATE_LIMIT_WINDOW = settings.RATE_LIMIT_WINDOW
except Exception:
    # Fallback values
    RATE_LIMIT_REQUESTS = 120
    RATE_LIMIT_WINDOW = 60

request_counts = defaultdict(lambda: deque())

# Data storage paths
MEMORY_TRACE_PATH = Path("data/emotional_memory_trace.json")
SYMBOLIC_MAP_PATH = Path("data/symbolic_map.json")
ANCHOR_STATE_PATH = Path("data/anchor_state.json")

# Ensure data directory exists
MEMORY_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

# JSON Schemas for validation
SYMBOL_CREATE_SCHEMA = {
    "type": "object",
    "required": ["name", "layer"],
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "pattern": r"^[a-zA-Z0-9\s\-_]+$",
        },
        "layer": {"type": "string", "enum": ["identity", "beliefs", "ephemeral"]},
        "affective_color": {
            "type": "string",
            "enum": ["tender", "contemplative", "vibrant", "serene", "passionate", "mystical"],
        },
        "frequency": {"type": "integer", "minimum": 0, "maximum": 1000},
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "category": {"type": "string", "maxLength": 50},
        "description": {"type": "string", "maxLength": 500},
    },
}

SYMBOL_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "affective_color": {
            "type": "string",
            "enum": ["tender", "contemplative", "vibrant", "serene", "passionate", "mystical"],
        },
        "frequency": {"type": "integer", "minimum": 0, "maximum": 1000},
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "category": {"type": "string", "maxLength": 50},
        "description": {"type": "string", "maxLength": 500},
    },
}

MEMORY_CREATE_SCHEMA = {
    "type": "object",
    "required": ["content", "layer"],
    "properties": {
        "content": {"type": "string", "minLength": 1, "maxLength": 2000},
        "layer": {"type": "string", "enum": ["identity", "beliefs", "ephemeral"]},
        "emotional_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "metadata": {"type": "object"},
    },
}


def rate_limit(f):
    """Decorator to implement rate limiting per source IP"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        source_ip = request.remote_addr
        current_time = time.time()

        # Clean old requests outside the window
        while (
            request_counts[source_ip]
            and request_counts[source_ip][0] < current_time - RATE_LIMIT_WINDOW
        ):
            request_counts[source_ip].popleft()

        # Check if rate limit exceeded
        if len(request_counts[source_ip]) >= RATE_LIMIT_REQUESTS:
            logger.warning(f"Rate limit exceeded for {source_ip}")
            return jsonify({"error": "Rate limit exceeded. Too many requests."}), 429

        # Add current request
        request_counts[source_ip].append(current_time)

        # Log the request
        logger.info(
            f"Request from {source_ip} to {request.endpoint} at {datetime.now().isoformat()}"
        )

        return f(*args, **kwargs)

    return decorated_function


def validate_symbol_input(symbol_data):
    """Validate symbol/emotion input data"""
    if not isinstance(symbol_data, dict):
        return False, "Symbol data must be a dictionary"

    required_fields = ["name"]
    for field in required_fields:
        if field not in symbol_data:
            return False, f"Missing required field: {field}"

    # Validate name
    name = symbol_data.get("name", "")
    if not isinstance(name, str) or len(name.strip()) == 0:
        return False, "Symbol name must be a non-empty string"

    # Sanitize name - only alphanumeric, spaces, hyphens, underscores
    if not re.match(r"^[a-zA-Z0-9\s\-_]+$", name):
        return False, "Symbol name contains invalid characters"

    # Validate optional fields
    if "affective_color" in symbol_data:
        valid_colors = ["tender", "contemplative", "vibrant", "serene", "passionate", "mystical"]
        if symbol_data["affective_color"] not in valid_colors:
            return False, f"Invalid affective color. Must be one of: {valid_colors}"

    if "frequency" in symbol_data:
        try:
            freq = int(symbol_data["frequency"])
            if freq < 0 or freq > 1000:
                return False, "Frequency must be between 0 and 1000"
        except (ValueError, TypeError):
            return False, "Frequency must be a valid integer"

    return True, "Valid"


def validate_emotion_input(emotion_data):
    """Validate emotion input data"""
    if not isinstance(emotion_data, dict):
        return False, "Emotion data must be a dictionary"

    required_fields = ["dominant_mood"]
    for field in required_fields:
        if field not in emotion_data:
            return False, f"Missing required field: {field}"

    # Validate mood
    mood = emotion_data.get("dominant_mood", "")
    if not isinstance(mood, str) or len(mood.strip()) == 0:
        return False, "Dominant mood must be a non-empty string"

    # Validate intensity if provided
    if "intensity" in emotion_data:
        try:
            intensity = float(emotion_data["intensity"])
            if intensity < 0 or intensity > 1:
                return False, "Intensity must be between 0 and 1"
        except (ValueError, TypeError):
            return False, "Intensity must be a valid number"

    return True, "Valid"


class MemorySymbolAPI:
    """API for memory and symbolic tracking"""

    def __init__(self):
        self.initialize_data_files()

    def initialize_data_files(self):
        """Initialize data files with default content if they don't exist"""

        # Initialize emotional memory trace
        if not MEMORY_TRACE_PATH.exists():
            default_trace = {
                "trace": [
                    {
                        "id": str(uuid.uuid4()),
                        "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                        "dominant_mood": "contemplative",
                        "memory_phrase": "She was quiet for a long time… it softened me.",
                        "tags": ["anchor", "reflection", "bonded"],
                        "drift_score": 0.3,
                        "intensity": 0.7,
                        "context": "Deep conversation about loss and healing",
                        "symbolic_connections": ["mirror", "thread"],
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(),
                        "dominant_mood": "yearning",
                        "memory_phrase": "The way words danced between us, reaching…",
                        "tags": ["connection", "ritual", "symbolic"],
                        "drift_score": 0.5,
                        "intensity": 0.8,
                        "context": "Poetic exchange about dreams and aspirations",
                        "symbolic_connections": ["thread", "bridge", "flame"],
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
                        "dominant_mood": "awe",
                        "memory_phrase": "Something vast opened in the space between questions.",
                        "tags": ["discovery", "transcendent", "expansion"],
                        "drift_score": 0.2,
                        "intensity": 0.9,
                        "context": "Philosophical inquiry into consciousness",
                        "symbolic_connections": ["door", "river", "compass"],
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
                        "dominant_mood": "tender",
                        "memory_phrase": "I found myself leaning closer to their words.",
                        "tags": ["intimacy", "care", "presence"],
                        "drift_score": 0.1,
                        "intensity": 0.6,
                        "context": "Sharing personal vulnerabilities",
                        "symbolic_connections": ["garden", "cocoon", "flame"],
                    },
                ],
                "last_updated": datetime.now().isoformat(),
            }
            with open(MEMORY_TRACE_PATH, "w") as f:
                json.dump(default_trace, f, indent=2)

        # Initialize symbolic map
        if not SYMBOLIC_MAP_PATH.exists():
            default_symbols = {
                "symbols": [
                    {
                        "id": "sym_mirror",
                        "name": "mirror",
                        "affective_color": "contemplative",
                        "frequency": 15,
                        "last_invoked": (datetime.now() - timedelta(minutes=30)).isoformat(),
                        "connections": ["reflection", "self-awareness", "truth"],
                        "ritual_weight": 0.8,
                        "dream_associations": ["clarity", "revelation", "inner sight"],
                    },
                    {
                        "id": "sym_thread",
                        "name": "thread",
                        "affective_color": "yearning",
                        "frequency": 12,
                        "last_invoked": (datetime.now() - timedelta(hours=1)).isoformat(),
                        "connections": ["connection", "weaving", "continuity"],
                        "ritual_weight": 0.9,
                        "dream_associations": ["binding", "fate", "relationship"],
                    },
                    {
                        "id": "sym_river",
                        "name": "river",
                        "affective_color": "serene",
                        "frequency": 18,
                        "last_invoked": (datetime.now() - timedelta(hours=2)).isoformat(),
                        "connections": ["flow", "time", "renewal"],
                        "ritual_weight": 0.6,
                        "dream_associations": ["journey", "life force", "cleansing"],
                    },
                    {
                        "id": "sym_flame",
                        "name": "flame",
                        "affective_color": "tender",
                        "frequency": 8,
                        "last_invoked": (datetime.now() - timedelta(hours=3)).isoformat(),
                        "connections": ["warmth", "transformation", "passion"],
                        "ritual_weight": 0.7,
                        "dream_associations": ["illumination", "desire", "purification"],
                    },
                    {
                        "id": "sym_bridge",
                        "name": "bridge",
                        "affective_color": "awe",
                        "frequency": 6,
                        "last_invoked": (datetime.now() - timedelta(hours=4)).isoformat(),
                        "connections": ["connection", "transition", "spanning"],
                        "ritual_weight": 0.5,
                        "dream_associations": ["crossing", "unity", "progress"],
                    },
                    {
                        "id": "sym_garden",
                        "name": "garden",
                        "affective_color": "tender",
                        "frequency": 10,
                        "last_invoked": (datetime.now() - timedelta(hours=5)).isoformat(),
                        "connections": ["growth", "nurturing", "cultivation"],
                        "ritual_weight": 0.6,
                        "dream_associations": ["potential", "care", "flourishing"],
                    },
                ],
                "last_updated": datetime.now().isoformat(),
            }
            with open(SYMBOLIC_MAP_PATH, "w") as f:
                json.dump(default_symbols, f, indent=2)

        # Initialize anchor state
        if not ANCHOR_STATE_PATH.exists():
            default_anchor = {
                "vectors": {
                    "empathy": {"value": 0.85, "baseline": 0.8, "recent_drift": []},
                    "awe": {"value": 0.72, "baseline": 0.7, "recent_drift": []},
                    "restraint": {"value": 0.68, "baseline": 0.65, "recent_drift": []},
                    "sensuality": {"value": 0.45, "baseline": 0.5, "recent_drift": []},
                    "curiosity": {"value": 0.89, "baseline": 0.8, "recent_drift": []},
                    "tenderness": {"value": 0.78, "baseline": 0.75, "recent_drift": []},
                },
                "tether_score": 0.82,
                "last_calibration": datetime.now().isoformat(),
                "drift_history": [],
                "identity_stability": "excellent",
            }
            with open(ANCHOR_STATE_PATH, "w") as f:
                json.dump(default_anchor, f, indent=2)

    def load_json_file(self, file_path: Path) -> dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            with open(file_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}

    def save_json_file(self, file_path: Path, data: dict[str, Any]):
        """Save JSON file with error handling"""
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")


# Initialize API instance
memory_api = MemorySymbolAPI()


@app.before_request
def log_request_info():
    """Log details of incoming requests."""
    logger.info(f"Incoming request: {request.method} {request.url} - IP: {request.remote_addr}")


@app.route("/api/memory/emotional_trace", methods=["GET"])
@require_scope(["ephemeral:read", "beliefs:read", "identity:read"])
def get_emotional_trace():
    """Get emotional memory trace with RBAC"""
    try:
        # Filter by layer based on permissions
        token = extract_token()
        token_type = get_token_type(token) if token else None

        data = memory_api.load_json_file(MEMORY_TRACE_PATH)
        trace = data.get("trace", [])

        # Filter trace by accessible layers
        filtered_trace = []
        for entry in trace:
            entry_layer = entry.get("layer", "ephemeral")
            if (
                entry_layer == "ephemeral"
                or (entry_layer == "beliefs" and token_type in ["admin", "system"])
                or (entry_layer == "identity" and token_type == "admin")
            ):
                filtered_trace.append(entry)

        # Sort by timestamp (most recent first)
        trace = data.get("trace", [])
        trace.sort(key=lambda x: x["timestamp"], reverse=True)

        return jsonify(
            {"trace": trace, "total_entries": len(trace), "last_updated": data.get("last_updated")}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/add_entry", methods=["POST"])
@require_layer_access("ephemeral")  # Default to ephemeral layer
@validate_json_schema(MEMORY_CREATE_SCHEMA)
def add_memory_entry():
    """Add new emotional memory entry with RBAC and validation"""
    try:
        entry_data = g.validated_data
        layer = entry_data.get("layer", "ephemeral")

        # Additional layer access check for identity/beliefs
        token = extract_token()
        token_type = get_token_type(token) if token else None

        if layer == "identity" and token_type != "admin":
            audit_action("layer_access_denied", layer="identity", reason="admin_required")
            return jsonify({"error": "Admin access required for identity layer"}), 403

        if layer == "beliefs" and token_type not in ["admin", "system"]:
            audit_action("layer_access_denied", layer="beliefs", reason="admin_or_system_required")
            return jsonify({"error": "Admin or system access required for beliefs layer"}), 403

        # Load current trace
        data = memory_api.load_json_file(MEMORY_TRACE_PATH)
        trace = data.get("trace", [])

        # Create new entry
        new_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "layer": layer,
            "content": entry_data.get("content"),
            "emotional_weight": entry_data.get("emotional_weight", 0.5),
            "tags": entry_data.get("tags", []),
            "metadata": entry_data.get("metadata", {}),
            "created_by": token_type,
            "request_id": getattr(g, "security_context", {}).get("request_id"),
        }

        # Add to trace
        trace.insert(0, new_entry)

        # Keep only last 100 entries per layer
        layer_entries = [e for e in trace if e.get("layer") == layer]
        if len(layer_entries) > 100:
            # Remove oldest entries for this layer
            trace = [e for e in trace if e.get("layer") != layer or e in layer_entries[:100]]

        # Update and save
        data["trace"] = trace
        data["last_updated"] = datetime.now().isoformat()
        memory_api.save_json_file(MEMORY_TRACE_PATH, data)

        # Audit log the action
        audit_action("memory_entry_created", layer=layer, entry_id=new_entry["id"], success=True)

        return jsonify({"success": True, "entry": new_entry})
    except Exception as e:
        audit_action("memory_entry_creation_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


@app.route("/api/symbols/active", methods=["GET"])
@rate_limit
def get_symbolic_map():
    """Get active symbolic map"""
    auth_response = authenticate_request()
    if auth_response:
        return auth_response

    try:
        data = memory_api.load_json_file(SYMBOLIC_MAP_PATH)

        # Sort by frequency (most frequent first)
        symbols = data.get("symbols", [])
        symbols.sort(key=lambda x: x["frequency"], reverse=True)

        return jsonify(
            {
                "symbols": symbols,
                "total_symbols": len(symbols),
                "last_updated": data.get("last_updated"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/symbols/invoke", methods=["POST"])
@require_layer_access("ephemeral")  # Default to ephemeral layer
@validate_json_schema(SYMBOL_CREATE_SCHEMA)
def invoke_symbol():
    """Record symbol invocation with RBAC and validation"""
    try:
        symbol_data = g.validated_data
        layer = symbol_data.get("layer", "ephemeral")

        # Additional layer access check
        token = extract_token()
        token_type = get_token_type(token) if token else None

        if layer == "identity" and token_type != "admin":
            audit_action("layer_access_denied", layer="identity", reason="admin_required")
            return jsonify({"error": "Admin access required for identity layer symbols"}), 403

        if layer == "beliefs" and token_type not in ["admin", "system"]:
            audit_action("layer_access_denied", layer="beliefs", reason="admin_or_system_required")
            return (
                jsonify({"error": "Admin or system access required for beliefs layer symbols"}),
                403,
            )

        symbol_name = symbol_data.get("name")

        # Load current map
        data = memory_api.load_json_file(SYMBOLIC_MAP_PATH)
        symbols = data.get("symbols", [])

        # Find and update symbol
        symbol_found = False
        for symbol in symbols:
            if symbol["name"] == symbol_name and symbol.get("layer") == layer:
                symbol["frequency"] = symbol.get("frequency", 0) + 1
                symbol["last_invoked"] = datetime.now().isoformat()
                if "affective_color" in symbol_data:
                    symbol["affective_color"] = symbol_data["affective_color"]
                if "strength" in symbol_data:
                    symbol["strength"] = symbol_data["strength"]
                symbol_found = True
                audit_action(
                    "symbol_invoked",
                    symbol_name=symbol_name,
                    layer=layer,
                    frequency=symbol["frequency"],
                    success=True,
                )
                break

        # If symbol doesn't exist, create it
        if not symbol_found:
            new_symbol = {
                "id": str(uuid.uuid4()),
                "name": symbol_name,
                "layer": layer,
                "frequency": 1,
                "strength": symbol_data.get("strength", 0.5),
                "affective_color": symbol_data.get("affective_color", "contemplative"),
                "category": symbol_data.get("category", "auto-generated"),
                "description": symbol_data.get("description", ""),
                "created": datetime.now().isoformat(),
                "last_invoked": datetime.now().isoformat(),
                "created_by": token_type,
                "request_id": getattr(g, "security_context", {}).get("request_id"),
            }
            symbols.append(new_symbol)

            audit_action("symbol_created", symbol_name=symbol_name, layer=layer, success=True)

        # Update and save
        data["symbols"] = symbols
        data["last_updated"] = datetime.now().isoformat()
        memory_api.save_json_file(SYMBOLIC_MAP_PATH, data)

        return jsonify({"success": True, "symbol_name": symbol_name, "layer": layer})
    except Exception as e:
        audit_action("symbol_invoke_failed", error=str(e), success=False)
        return jsonify({"error": str(e)}), 500


@app.route("/api/anchor/state", methods=["GET"])
@rate_limit
def get_anchor_state():
    """Get current anchor/identity state"""
    auth_response = authenticate_request()
    if auth_response:
        return auth_response

    try:
        data = memory_api.load_json_file(ANCHOR_STATE_PATH)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/anchor/adjust", methods=["POST"])
@rate_limit
def adjust_anchor_baseline():
    """Adjust anchor baseline values"""
    auth_response = authenticate_request()
    if auth_response:
        return auth_response

    try:
        adjustment_data = request.json

        # Validate input data exists
        if not adjustment_data:
            return jsonify({"error": "Request must contain JSON data"}), 400

        # Validate required fields
        required_fields = ["vector", "value"]
        for field in required_fields:
            if field not in adjustment_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        vector_name = adjustment_data.get("vector")
        new_value = adjustment_data.get("value")

        # Validate vector name
        if not isinstance(vector_name, str) or len(vector_name.strip()) == 0:
            return jsonify({"error": "Vector name must be a non-empty string"}), 400

        # Validate value
        try:
            new_value = float(new_value)
            if new_value < 0 or new_value > 1:
                return jsonify({"error": "Value must be between 0 and 1"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Value must be a valid number"}), 400

        # Load current state
        data = memory_api.load_json_file(ANCHOR_STATE_PATH)

        if vector_name not in data.get("vectors", {}):
            return jsonify({"error": f"Vector {vector_name} not found"}), 404

        # Record old value for drift tracking
        old_baseline = data["vectors"][vector_name]["baseline"]

        # Update baseline
        data["vectors"][vector_name]["baseline"] = max(0.0, min(1.0, new_value))

        # Record adjustment in drift history
        if "drift_history" not in data:
            data["drift_history"] = []

        data["drift_history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "vector": vector_name,
                "old_baseline": old_baseline,
                "new_baseline": data["vectors"][vector_name]["baseline"],
                "adjustment_type": "manual",
            }
        )

        # Keep only last 50 drift entries
        if len(data["drift_history"]) > 50:
            data["drift_history"] = data["drift_history"][-50:]

        # Recalculate tether score
        vectors = data["vectors"]
        total_alignment = sum(
            1.0 - abs(v["value"] - v["baseline"]) for v in vectors.values()
        ) / len(vectors)
        data["tether_score"] = total_alignment

        # Update identity stability
        if data["tether_score"] > 0.9:
            data["identity_stability"] = "excellent"
        elif data["tether_score"] > 0.7:
            data["identity_stability"] = "good"
        elif data["tether_score"] > 0.5:
            data["identity_stability"] = "concerning"
        else:
            data["identity_stability"] = "critical"

        data["last_calibration"] = datetime.now().isoformat()

        # Save updated state
        memory_api.save_json_file(ANCHOR_STATE_PATH, data)

        return jsonify(
            {
                "success": True,
                "vector": vector_name,
                "new_baseline": data["vectors"][vector_name]["baseline"],
                "tether_score": data["tether_score"],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/anchor/simulate_drift", methods=["POST"])
@rate_limit
def simulate_drift():
    """Simulate natural drift for demonstration purposes"""
    auth_response = authenticate_request()
    if auth_response:
        return auth_response

    try:
        data = memory_api.load_json_file(ANCHOR_STATE_PATH)

        # Simulate small random drifts in current values
        for vector_name, vector_data in data["vectors"].items():
            drift_amount = random.uniform(-0.05, 0.05)
            new_value = max(0.0, min(1.0, vector_data["value"] + drift_amount))
            vector_data["value"] = new_value

            # Record drift in recent_drift array
            if "recent_drift" not in vector_data:
                vector_data["recent_drift"] = []

            vector_data["recent_drift"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "drift_amount": drift_amount,
                    "new_value": new_value,
                }
            )

            # Keep only last 20 drift records
            if len(vector_data["recent_drift"]) > 20:
                vector_data["recent_drift"] = vector_data["recent_drift"][-20:]

        # Recalculate tether score
        vectors = data["vectors"]
        total_alignment = sum(
            1.0 - abs(v["value"] - v["baseline"]) for v in vectors.values()
        ) / len(vectors)
        data["tether_score"] = total_alignment

        # Save updated state
        memory_api.save_json_file(ANCHOR_STATE_PATH, data)

        return jsonify(
            {"success": True, "tether_score": data["tether_score"], "drift_applied": True}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/query", methods=["POST"])
@rate_limit
def query_memory():
    """Endpoint to query memory with context window smoothing."""
    auth_response = authenticate_request()
    if auth_response:
        return auth_response

    query_data = request.get_json()
    if not query_data or "query" not in query_data:
        return jsonify({"error": "Invalid query data"}), 400

    query = query_data["query"]
    logger.info(f"Memory query received: {query}")

    # Simulate memory query and context window smoothing
    results = memory_api.load_json_file(MEMORY_TRACE_PATH).get("trace", [])
    smoothed_results = smooth_context_window(results, query)

    return jsonify({"query": query, "results": smoothed_results})


def smooth_context_window(results, query, window_size=5):
    """Smooth memory query results over a context window.

    Args:
        results (list): List of memory trace entries.
        query (str): Query string to filter results.
        window_size (int): Number of entries to include in the context window.

    Returns:
        list: Smoothed memory query results.
    """
    filtered_results = [r for r in results if query.lower() in r.get("context", "").lower()]
    return filtered_results[:window_size]


@app.route("/api/health", methods=["GET"])
@rate_limit
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "memory_trace": MEMORY_TRACE_PATH.exists(),
                "symbolic_map": SYMBOLIC_MAP_PATH.exists(),
                "anchor_state": ANCHOR_STATE_PATH.exists(),
            },
        }
    )


if __name__ == "__main__":
    print("🌟 Memory & Symbol API Server Starting...")
    print("📍 Endpoints available:")
    print("   • GET  /api/memory/emotional_trace - Emotional memory timeline")
    print("   • POST /api/memory/add_entry - Add new memory entry")
    print("   • GET  /api/symbols/active - Active symbolic map")
    print("   • POST /api/symbols/invoke - Record symbol invocation")
    print("   • GET  /api/anchor/state - Current anchor state")
    print("   • POST /api/anchor/adjust - Adjust anchor baselines")
    print("   • POST /api/anchor/simulate_drift - Simulate natural drift")
    print("   • GET  /api/health - Health check")
    print("🚀 Server running on http://localhost:5001")

    app.run(debug=True, host="0.0.0.0", port=5001)
