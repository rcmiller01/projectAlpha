#!/usr/bin/env python3
"""
CoreArbiter API Integration

Flask API endpoints for integrating CoreArbiter with the existing system.
Enhanced with RBAC, token masking, and comprehensive audit logging.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, Response, g, jsonify, request
from flask_cors import CORS

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.common.retry import NetworkError, ServiceUnavailableError, retry_arbiter_call

# Import security utilities
from backend.common.security import (
    audit_action,
    create_request_context,
    extract_token,
    get_token_type,
    mask_token,
    require_scope,
    validate_json_schema,
)
from core.core_arbiter import CoreArbiter, WeightingStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global CoreArbiter instance
core_arbiter = None

# Idempotency cache for mutating operations
idempotency_cache = {}
IDEMPOTENCY_CACHE_TTL = 3600  # 1 hour

def cleanup_idempotency_cache():
    """Clean up expired idempotency cache entries."""
    import time
    current_time = time.time()
    expired_keys = [
        key for key, (_, timestamp) in idempotency_cache.items()
        if current_time - timestamp > IDEMPOTENCY_CACHE_TTL
    ]
    for key in expired_keys:
        del idempotency_cache[key]

def check_idempotency(idempotency_key: str) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check if request is idempotent and return cached response if available.

    Args:
        idempotency_key: Unique key for this operation

    Returns:
        Tuple of (is_duplicate, cached_response)
    """
    if not idempotency_key:
        return False, None

    cleanup_idempotency_cache()

    if idempotency_key in idempotency_cache:
        cached_response, timestamp = idempotency_cache[idempotency_key]
        logger.info(f"Returning cached response for idempotency key: {mask_token(idempotency_key)}")
        return True, cached_response

    return False, None

def store_idempotency_response(idempotency_key: str, response: Dict[str, Any]):
    """Store response in idempotency cache."""
    if idempotency_key:
        import time
        idempotency_cache[idempotency_key] = (response, time.time())

def require_idempotency(func):
    """Decorator to handle idempotency for mutating operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get idempotency key from header
        idempotency_key = request.headers.get('Idempotency-Key')

        if idempotency_key:
            # Check if this is a duplicate request
            is_duplicate, cached_response = check_idempotency(idempotency_key)
            if is_duplicate:
                audit_action(
                    route=request.endpoint or request.path,
                    action="idempotent_duplicate",
                    success=True,
                    details={"idempotency_key": mask_token(idempotency_key)}
                )
                return jsonify(cached_response)

        # Execute the original function
        response = func(*args, **kwargs)

        # Store response in cache if idempotency key provided
        if idempotency_key and hasattr(response, 'get_json'):
            response_data = response.get_json()
            if response_data and response.status_code == 200:
                store_idempotency_response(idempotency_key, response_data)

        return response
    return wrapper

# JSON Schemas for validation
PROCESS_INPUT_SCHEMA = {
    "type": "object",
    "required": ["message"],
    "properties": {
        "message": {
            "type": "string",
            "minLength": 1,
            "maxLength": 5000
        },
        "state": {
            "type": "object"
        },
        "context": {
            "type": "object"
        },
        "options": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["balanced", "emotional", "logical", "creative"]
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300
                }
            }
        }
    }
}

@app.before_request
def log_request_with_masked_tokens():
    """Log requests with properly masked tokens."""
    token = extract_token()
    masked_token = mask_token(token)

    logger.info(f"API Request: {request.method} {request.endpoint} - "
               f"IP: {request.remote_addr} - Token: {masked_token}")

    # Create audit context
    audit_action('api_request',
                endpoint=request.endpoint,
                method=request.method,
                token_type=get_token_type(token) if token else None)

def get_arbiter():
    """Get or create CoreArbiter instance"""
    global core_arbiter
    if core_arbiter is None:
        core_arbiter = CoreArbiter()
    return core_arbiter

@app.route('/api/arbiter/process', methods=['POST'])
@require_scope(['system:operate', 'user:basic'])
@validate_json_schema(PROCESS_INPUT_SCHEMA)
def process_input() -> Response:
    """Process user input through CoreArbiter with RBAC"""
    try:
        # Get validated data
        data = g.validated_data

        user_input = data.get('message', '')
        state = data.get('state', {})
        context = data.get('context', {})
        options = data.get('options', {})

        # Audit log the processing request
        audit_action('arbiter_process_input',
                    input_length=len(user_input),
                    has_state=bool(state),
                    has_context=bool(context),
                    success=True)

        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            arbiter = get_arbiter()
            response = loop.run_until_complete(arbiter.process_input(user_input, state))

            # Convert response to JSON-serializable format
            response_data = {
                'final_output': response.final_output,
                'reflection': response.reflection,
                'action': response.action,
                'mood_inflected': response.mood_inflected,
                'tone': response.tone,
                'priority': response.priority,
                'source_weights': response.source_weights,
                'confidence': response.confidence,
                'emotional_override': response.emotional_override,
                'symbolic_context': response.symbolic_context,
                'resolution_strategy': response.resolution_strategy,
                'timestamp': response.timestamp.isoformat(),
                'metadata': response.metadata
            }

            return jsonify(response_data)

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error processing input: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/arbiter/status', methods=['GET'])
@require_scope(['system:operate', 'user:basic'])
def get_status() -> Response:
    """Get current arbiter system status with RBAC"""
    try:
        arbiter = get_arbiter()
        status = arbiter.get_system_status()

        audit_action('arbiter_status_request', success=True)
        return jsonify(status)
    except Exception as e:
        audit_action('arbiter_status_failed', error=str(e), success=False)
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/arbiter/strategy', methods=['POST'])
@require_scope(['system:operate'])
@require_idempotency
@validate_json_schema({
    "type": "object",
    "required": ["strategy"],
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["balanced", "emotional", "logical", "creative"]
        }
    }
})
def set_strategy() -> Response:
    """Change weighting strategy (system access required)"""
    try:
        data = g.validated_data
        strategy_name = data.get('strategy')

        # Map strategy names to enum values (simplified for now)
        strategy_map = {
            'balanced': 'balanced',
            'emotional': 'emotional',
            'logical': 'logical',
            'creative': 'creative'
        }

        if strategy_name not in strategy_map:
            audit_action('arbiter_strategy_invalid', strategy=strategy_name, success=False)
            return jsonify({'error': f'Invalid strategy: {strategy_name}'}), 400

        arbiter = get_arbiter()
        strategy = strategy_map[strategy_name]
        arbiter.set_weighting_strategy(strategy)

        audit_action('arbiter_strategy_changed',
                    old_strategy='unknown',
                    new_strategy=strategy_name,
                    success=True)

        return jsonify({
            'success': True,
            'strategy': strategy_name,
            'message': f'Strategy changed to {strategy_name}'
        })
    except Exception as e:
        audit_action('arbiter_strategy_failed', error=str(e), success=False)
        logger.error(f"Error setting strategy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/arbiter/regulate', methods=['POST'])
def regulate_system() -> Response:
    """Perform system regulation"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            arbiter = get_arbiter()
            loop.run_until_complete(arbiter.regulate_system())

            # Get updated status
            status = arbiter.get_system_status()

            return jsonify({
                'status': 'success',
                'message': 'System regulation completed',
                'system_status': status
            })

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error regulating system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotional_state', methods=['GET'])
def get_emotional_state() -> Response:
    """Get current emotional state for UI"""
    try:
        # Load from file or generate current state
        emotional_state_path = Path("data/emotional_state.json")

        if emotional_state_path.exists():
            with open(emotional_state_path, 'r') as f:
                state = json.load(f)
        else:
            # Generate default state
            state = {
                "valence": 0.2,
                "arousal": 0.4,
                "dominant_emotion": "contemplative",
                "stability": 0.85,
                "mood_signals": {
                    "warmth": 0.7,
                    "empathy": 0.8,
                    "curiosity": 0.6,
                    "concern": 0.3
                }
            }

        # Add arbiter status if available
        if core_arbiter:
            arbiter_status = core_arbiter.get_system_status()
            state['arbiter_status'] = arbiter_status

        return jsonify(state)

    except Exception as e:
        logger.error(f"Error getting emotional state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbolic_response', methods=['POST'])
def generate_symbolic_response() -> Response:
    """Generate symbolic/ritual response"""
    try:
        data = request.json
        current_state = data.get('current_state', {})
        context = data.get('context', [])

        # Create symbolic input for arbiter
        symbolic_input = "Express the deeper symbolic meaning of our connection"
        state = {
            'context': 'symbolic_expression',
            'emotional_state': current_state,
            'recent_context': context,
            'ritual_request': True
        }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            arbiter = get_arbiter()
            response = loop.run_until_complete(arbiter.process_input(symbolic_input, state))

            return jsonify({
                'symbolic_output': response.final_output,
                'reflection': response.reflection,
                'symbolic_context': response.symbolic_context,
                'ritual_strength': response.symbolic_context.get('ritual_strength', 0.5)
            })

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error generating symbolic response: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/log_emotional_message', methods=['POST'])
def log_emotional_message() -> Response:
    """Log message with emotional context"""
    try:
        data = request.json

        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': data.get('message'),
            'role': data.get('role'),
            'emotional_state': data.get('emotional_state'),
            'mood_profile': data.get('mood_profile')
        }

        # Append to emotional conversation log
        log_path = Path("logs/emotional_conversations.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_path.exists():
            with open(log_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]

        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)

        return jsonify({'status': 'logged'})

    except Exception as e:
        logger.error(f"Error logging message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/arbiter/traces', methods=['GET'])
def get_traces() -> Response:
    """Get arbiter decision traces"""
    try:
        trace_path = Path("logs/core_arbiter_trace.json")

        if trace_path.exists():
            with open(trace_path, 'r') as f:
                traces = json.load(f)

            # Return last N traces
            limit = request.args.get('limit', 50, type=int)
            return jsonify({
                'traces': traces[-limit:],
                'total_count': len(traces)
            })
        else:
            return jsonify({
                'traces': [],
                'total_count': 0
            })

    except Exception as e:
        logger.error(f"Error getting traces: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_arbiter() -> Response:
    """Main chat endpoint using CoreArbiter"""
    try:
        data = request.json
        message = data.get('message', '')
        emotional_context = data.get('emotional_context', {})
        mood_profile = data.get('mood_profile', {})

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Prepare state for arbiter
        state = {
            'context': 'conversational_chat',
            'emotional_context': emotional_context,
            'mood_profile': mood_profile,
            'user_message': message
        }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            arbiter = get_arbiter()
            response = loop.run_until_complete(arbiter.process_input(message, state))

            # Generate mood profile based on response
            mood_colors = {
                'emotional': {'primary': '#EC4899', 'secondary': '#F9A8D4'},
                'balanced': {'primary': '#8B5CF6', 'secondary': '#C4B5FD'},
                'objective': {'primary': '#06B6D4', 'secondary': '#67E8F9'}
            }

            colors = mood_colors.get(response.tone, mood_colors['balanced'])

            response_mood = {
                'emotion': response.symbolic_context.get('mood_primary', 'contemplative'),
                'intensity': response.confidence,
                'colors': colors,
                'icon': '🤔' if response.tone == 'objective' else '💭' if response.tone == 'balanced' else '💖'
            }

            return jsonify({
                'response': response.final_output,
                'mood_profile': response_mood,
                'metadata': {
                    'confidence': response.confidence,
                    'tone': response.tone,
                    'priority': response.priority,
                    'emotional_override': response.emotional_override,
                    'resolution_strategy': response.resolution_strategy,
                    'symbolic_context': response.symbolic_context
                },
                'reflection': response.reflection,
                'timestamp': response.timestamp.isoformat()
            })

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check() -> Response:
    """Health check endpoint"""
    try:
        status = "healthy"
        if core_arbiter:
            arbiter_status = core_arbiter.get_system_status()
            if arbiter_status['health_status'] in ['critical', 'concerning']:
                status = "degraded"

        return jsonify({
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'arbiter_initialized': core_arbiter is not None
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Ensure directories exist
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Initialize arbiter
    get_arbiter()

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
