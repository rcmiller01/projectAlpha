"""
Common security utilities for ProjectAlpha backend APIs.
Provides RBAC, token validation, audit logging, and security helpers.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from flask import request, jsonify, g

logger = logging.getLogger(__name__)

# Security configuration
ADMIN_TOKEN_PREFIX = "admin_"
SYSTEM_TOKEN_PREFIX = "sys_"
USER_TOKEN_PREFIX = "user_"

# Audit log configuration
AUDIT_LOG_PATH = Path("logs/audit.jsonl")
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Token scopes and permissions
TOKEN_SCOPES = {
    'admin': {'identity:read', 'identity:write', 'beliefs:read', 'beliefs:write', 'ephemeral:read', 'ephemeral:write', 'system:admin'},
    'system': {'beliefs:read', 'beliefs:write', 'ephemeral:read', 'ephemeral:write', 'system:operate'},
    'user': {'ephemeral:read', 'ephemeral:write', 'user:basic'},
}

# Layer protection mappings
LAYER_PERMISSIONS = {
    'identity': {'admin'},
    'beliefs': {'admin', 'system'},
    'ephemeral': {'admin', 'system', 'user'},
}

def mask_token(token: Optional[str]) -> str:
    """
    Mask token for logging, showing only last 4 characters.
    
    Args:
        token: The token to mask
        
    Returns:
        Masked token string
    """
    if not token:
        return "none"
    if len(token) <= 4:
        return "***"
    return f"***{token[-4:]}"

def get_token_type(token: str) -> Optional[str]:
    """
    Determine token type from token string.
    
    Args:
        token: The token to analyze
        
    Returns:
        Token type ('admin', 'system', 'user') or None
    """
    if not token:
        return None
    
    if token.startswith(ADMIN_TOKEN_PREFIX):
        return 'admin'
    elif token.startswith(SYSTEM_TOKEN_PREFIX):
        return 'system'
    elif token.startswith(USER_TOKEN_PREFIX):
        return 'user'
    
    # Legacy token checking
    admin_key = os.getenv('ADMIN_MASTER_KEY')
    if admin_key and token == admin_key:
        return 'admin'
    
    return None

def is_admin(token: str) -> bool:
    """
    Check if token has admin privileges.
    
    Args:
        token: The token to check
        
    Returns:
        True if token is admin
    """
    return get_token_type(token) == 'admin'

def get_token_scopes(token: str) -> Set[str]:
    """
    Get scopes for a given token.
    
    Args:
        token: The token to check
        
    Returns:
        Set of scopes for the token
    """
    token_type = get_token_type(token)
    if token_type is None:
        return set()
    return TOKEN_SCOPES.get(token_type, set())

def has_scope(token: str, required_scope: str) -> bool:
    """
    Check if token has required scope.
    
    Args:
        token: The token to check
        required_scope: The scope to check for
        
    Returns:
        True if token has scope
    """
    token_scopes = get_token_scopes(token)
    return required_scope in token_scopes

def can_access_layer(token: str, layer: str) -> bool:
    """
    Check if token can access specified layer.
    
    Args:
        token: The token to check
        layer: The layer to check access for
        
    Returns:
        True if access is allowed
    """
    token_type = get_token_type(token)
    if not token_type:
        return False
    
    allowed_types = LAYER_PERMISSIONS.get(layer, set())
    return token_type in allowed_types

def extract_token() -> Optional[str]:
    """
    Extract token from request headers or query parameters.
    
    Returns:
        Token string or None
    """
    # Check Authorization header
    auth_header = request.headers.get('Authorization')
    if auth_header:
        if auth_header.startswith('Bearer '):
            return auth_header[7:]
        elif auth_header.startswith('Token '):
            return auth_header[6:]
    
    # Check X-API-Key header
    api_key = request.headers.get('X-API-Key')
    if api_key:
        return api_key
    
    # Check query parameter
    return request.args.get('token') or request.args.get('api_key')

def require_scope(required_scopes: List[str]):
    """
    Decorator to require specific scopes for API access.
    
    Args:
        required_scopes: List of required scopes (OR logic)
        
    Returns:
        Decorator function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = extract_token()
            request_id = str(uuid.uuid4())
            
            # Log access attempt
            audit_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': request_id,
                'route': request.endpoint,
                'method': request.method,
                'actor': mask_token(token),
                'action': 'access_attempt',
                'required_scopes': required_scopes,
                'success': False,
                'ip': request.remote_addr,
            }
            
            if not token:
                audit_entry['error'] = 'no_token'
                log_audit_entry(audit_entry)
                return jsonify({'error': 'Authentication required', 'request_id': request_id}), 401
            
            # Check if token has any of the required scopes
            token_scopes = get_token_scopes(token)
            has_required_scope = any(has_scope(token, scope) for scope in required_scopes)
            
            if not has_required_scope:
                audit_entry['error'] = 'insufficient_scope'
                audit_entry['token_scopes'] = list(token_scopes)
                log_audit_entry(audit_entry)
                return jsonify({
                    'error': 'Insufficient permissions',
                    'required_scopes': required_scopes,
                    'request_id': request_id
                }), 403
            
            # Success
            audit_entry['success'] = True
            audit_entry['token_type'] = get_token_type(token)
            log_audit_entry(audit_entry)
            
            # Add request context for use in the endpoint
            g.security_context = {
                'token': token,
                'token_type': get_token_type(token),
                'scopes': token_scopes,
                'request_id': request_id,
            }
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_layer_access(layer: str):
    """
    Decorator to require access to specific layer.
    
    Args:
        layer: Layer name to protect
        
    Returns:
        Decorator function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = extract_token()
            request_id = str(uuid.uuid4())
            
            # Log access attempt
            audit_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': request_id,
                'route': request.endpoint,
                'method': request.method,
                'actor': mask_token(token),
                'action': 'layer_access',
                'layer': layer,
                'success': False,
                'ip': request.remote_addr,
            }
            
            if not token:
                audit_entry['error'] = 'no_token'
                log_audit_entry(audit_entry)
                return jsonify({'error': 'Authentication required', 'request_id': request_id}), 401
            
            if not can_access_layer(token, layer):
                audit_entry['error'] = 'layer_access_denied'
                audit_entry['token_type'] = get_token_type(token)
                log_audit_entry(audit_entry)
                return jsonify({
                    'error': f'Access denied to {layer} layer',
                    'required_permission': LAYER_PERMISSIONS.get(layer, []),
                    'request_id': request_id
                }), 403
            
            # Success
            audit_entry['success'] = True
            audit_entry['token_type'] = get_token_type(token)
            log_audit_entry(audit_entry)
            
            # Add request context
            g.security_context = {
                'token': token,
                'token_type': get_token_type(token),
                'layer': layer,
                'request_id': request_id,
            }
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_json_schema(schema: Dict[str, Any], allow_unknown: bool = False):
    """
    Decorator to validate JSON payload against schema.
    
    Args:
        schema: JSON schema definition
        allow_unknown: Whether to allow unknown fields
        
    Returns:
        Decorator function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            request_id = getattr(g, 'security_context', {}).get('request_id', str(uuid.uuid4()))
            
            if not request.is_json:
                audit_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'request_id': request_id,
                    'route': request.endpoint,
                    'method': request.method,
                    'action': 'validation_error',
                    'error': 'not_json',
                    'success': False,
                }
                log_audit_entry(audit_entry)
                return jsonify({'error': 'Content-Type must be application/json', 'request_id': request_id}), 400
            
            try:
                data = request.get_json()
            except Exception as e:
                audit_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'request_id': request_id,
                    'route': request.endpoint,
                    'method': request.method,
                    'action': 'validation_error',
                    'error': 'invalid_json',
                    'details': str(e),
                    'success': False,
                }
                log_audit_entry(audit_entry)
                return jsonify({'error': 'Invalid JSON payload', 'request_id': request_id}), 400
            
            # Basic schema validation
            validation_errors = validate_payload(data, schema, allow_unknown)
            if validation_errors:
                audit_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'request_id': request_id,
                    'route': request.endpoint,
                    'method': request.method,
                    'action': 'validation_error',
                    'error': 'schema_validation',
                    'validation_errors': validation_errors,
                    'success': False,
                }
                log_audit_entry(audit_entry)
                return jsonify({
                    'error': 'Validation failed',
                    'validation_errors': validation_errors,
                    'request_id': request_id
                }), 400
            
            # Add validated data to request context
            g.validated_data = data
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_payload(data: Any, schema: Dict[str, Any], allow_unknown: bool = False) -> List[str]:
    """
    Validate payload against simple schema.
    
    Args:
        data: Data to validate
        schema: Schema definition
        allow_unknown: Whether to allow unknown fields
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if not isinstance(data, dict):
        return ['Payload must be a JSON object']
    
    # Check required fields
    required_fields = schema.get('required', [])
    for field in required_fields:
        if field not in data:
            errors.append(f'Missing required field: {field}')
    
    # Check field types and constraints
    properties = schema.get('properties', {})
    for field, value in data.items():
        if field not in properties:
            if not allow_unknown:
                errors.append(f'Unknown field: {field}')
            continue
        
        field_schema = properties[field]
        field_type = field_schema.get('type')
        
        # Type checking
        if field_type == 'string' and not isinstance(value, str):
            errors.append(f'Field {field} must be a string')
        elif field_type == 'integer' and not isinstance(value, int):
            errors.append(f'Field {field} must be an integer')
        elif field_type == 'number' and not isinstance(value, (int, float)):
            errors.append(f'Field {field} must be a number')
        elif field_type == 'boolean' and not isinstance(value, bool):
            errors.append(f'Field {field} must be a boolean')
        elif field_type == 'array' and not isinstance(value, list):
            errors.append(f'Field {field} must be an array')
        elif field_type == 'object' and not isinstance(value, dict):
            errors.append(f'Field {field} must be an object')
        
        # String constraints
        if field_type == 'string' and isinstance(value, str):
            min_length = field_schema.get('minLength')
            max_length = field_schema.get('maxLength')
            pattern = field_schema.get('pattern')
            
            if min_length and len(value) < min_length:
                errors.append(f'Field {field} must be at least {min_length} characters')
            if max_length and len(value) > max_length:
                errors.append(f'Field {field} must be at most {max_length} characters')
            if pattern and not re.match(pattern, value):
                errors.append(f'Field {field} does not match required pattern')
        
        # Enum validation
        enum_values = field_schema.get('enum')
        if enum_values and value not in enum_values:
            errors.append(f'Field {field} must be one of: {enum_values}')
    
    return errors

def log_audit_entry(entry: Dict[str, Any]) -> None:
    """
    Log audit entry to append-only audit log.
    
    Args:
        entry: Audit entry to log
    """
    try:
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            json.dump(entry, f, separators=(',', ':'))
            f.write('\n')
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")

def audit_action(action: str, **kwargs) -> None:
    """
    Log an audit action with context.
    
    Args:
        action: Action description
        **kwargs: Additional audit data
    """
    request_id = getattr(g, 'security_context', {}).get('request_id', str(uuid.uuid4()))
    token = getattr(g, 'security_context', {}).get('token')
    
    audit_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'request_id': request_id,
        'route': request.endpoint,
        'method': request.method,
        'actor': mask_token(token),
        'action': action,
        'ip': request.remote_addr,
        **kwargs
    }
    
    log_audit_entry(audit_entry)

def create_request_context() -> Dict[str, Any]:
    """
    Create request context for logging.
    
    Returns:
        Request context dictionary
    """
    return {
        'request_id': str(uuid.uuid4()),
        'timestamp': datetime.utcnow().isoformat(),
        'route': request.endpoint,
        'method': request.method,
        'ip': request.remote_addr,
    }
