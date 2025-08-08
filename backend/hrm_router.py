"""
HRM Router for managing routing between different HRM subsystems.
Provides centralized routing with security enforcement.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.security import (
    extract_token, get_token_type, can_access_layer,
    audit_action, log_audit_entry
)

logger = logging.getLogger(__name__)

# Idempotency cache for HRM operations
hrm_idempotency_cache = {}
HRM_CACHE_TTL = 1800  # 30 minutes

def cleanup_hrm_cache():
    """Clean up expired HRM idempotency cache entries."""
    import time
    current_time = time.time()
    expired_keys = [
        key for key, (_, timestamp) in hrm_idempotency_cache.items()
        if current_time - timestamp > HRM_CACHE_TTL
    ]
    for key in expired_keys:
        del hrm_idempotency_cache[key]

class HRMRouter:
    """
    Centralized router for HRM system with layer-based access control.
    """

    def __init__(self):
        self.config = self._load_config()
        self.layer_handlers = {
            'identity': self._handle_identity_request,
            'beliefs': self._handle_beliefs_request,
            'ephemeral': self._handle_ephemeral_request,
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load HRM configuration."""
        try:
            from config.settings import get_settings
            settings = get_settings()
            return {
                'rate_limit_window': settings.RATE_LIMIT_WINDOW,
                'rate_limit_max': settings.RATE_LIMIT_MAX,
                'safe_mode': settings.SAFE_MODE_FORCE,
                'debug': settings.DEBUG,
            }
        except Exception:
            # Fallback configuration
            return {
                'rate_limit_window': 60,
                'rate_limit_max': 120,
                'safe_mode': False,
                'debug': False,
            }

    def route_request(self, layer: str, operation: str, data: Optional[Dict[str, Any]] = None,
                     token: Optional[str] = None) -> Dict[str, Any]:
        """
        Route request to appropriate layer handler with security checks.

        Args:
            layer: Target layer (identity, beliefs, ephemeral)
            operation: Operation type (read, write, update, delete)
            data: Request data
            token: Authentication token

        Returns:
            Response dictionary
        """
        request_id = f"hrm_{datetime.utcnow().timestamp()}"

        # Audit log the request
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request_id,
            'component': 'hrm_router',
            'action': f'{layer}_{operation}',
            'actor': token[-4:] if token and len(token) > 4 else 'unknown',
            'layer': layer,
            'operation': operation,
        }

        try:
            # Validate layer
            if layer not in self.layer_handlers:
                audit_entry['success'] = 'false'
                audit_entry['error'] = 'invalid_layer'
                log_audit_entry(audit_entry)
                return {'error': f'Invalid layer: {layer}', 'request_id': request_id}

            # Check access permissions
            if not token or not can_access_layer(token, layer):
                audit_entry['success'] = 'false'
                audit_entry['error'] = 'access_denied'
                log_audit_entry(audit_entry)
                return {'error': f'Access denied to {layer} layer', 'request_id': request_id}

            # Route to handler
            handler = self.layer_handlers[layer]
            result = handler(operation, data, token, request_id)

            audit_entry['success'] = 'true'
            log_audit_entry(audit_entry)

            return result

        except Exception as e:
            audit_entry['success'] = 'false'
            audit_entry['error'] = str(e)
            log_audit_entry(audit_entry)
            return {'error': f'Internal error: {str(e)}', 'request_id': request_id}

    def _handle_identity_request(self, operation: str, data: Optional[Dict[str, Any]],
                                token: Optional[str], request_id: str) -> Dict[str, Any]:
        """Handle identity layer requests (admin only)."""
        token_type = get_token_type(token) if token else None

        if token_type != 'admin':
            return {'error': 'Admin access required for identity layer', 'request_id': request_id}

        # Load identity data
        identity_path = Path("data/identity_layer.json")

        if operation == 'read':
            try:
                if identity_path.exists():
                    with open(identity_path, 'r') as f:
                        identity_data = json.load(f)
                else:
                    identity_data = {'data': {}, 'version': 1}

                return {'success': True, 'data': identity_data, 'request_id': request_id}
            except Exception as e:
                return {'error': f'Failed to read identity: {e}', 'request_id': request_id}

        elif operation in ['write', 'update']:
            if not data:
                return {'error': 'Data required for write operation', 'request_id': request_id}

            try:
                # Load existing data
                if identity_path.exists():
                    with open(identity_path, 'r') as f:
                        identity_data = json.load(f)
                else:
                    identity_data = {'data': {}, 'version': 1}

                # Update data
                if operation == 'write':
                    identity_data['data'] = data
                else:  # update
                    identity_data['data'].update(data)

                identity_data['last_updated'] = datetime.utcnow().isoformat()
                identity_data['version'] = identity_data.get('version', 1) + 1
                identity_data['updated_by'] = request_id

                # Save data
                identity_path.parent.mkdir(parents=True, exist_ok=True)
                with open(identity_path, 'w') as f:
                    json.dump(identity_data, f, indent=2)

                return {'success': True, 'message': 'Identity updated', 'request_id': request_id}
            except Exception as e:
                return {'error': f'Failed to update identity: {e}', 'request_id': request_id}

        return {'error': f'Unsupported operation: {operation}', 'request_id': request_id}

    def _handle_beliefs_request(self, operation: str, data: Optional[Dict[str, Any]],
                               token: Optional[str], request_id: str) -> Dict[str, Any]:
        """Handle beliefs layer requests (admin/system)."""
        token_type = get_token_type(token) if token else None

        if token_type not in ['admin', 'system']:
            return {'error': 'Admin or system access required for beliefs layer', 'request_id': request_id}

        # Load beliefs data
        beliefs_path = Path("data/beliefs_layer.json")

        if operation == 'read':
            try:
                if beliefs_path.exists():
                    with open(beliefs_path, 'r') as f:
                        beliefs_data = json.load(f)
                else:
                    beliefs_data = {'data': {}, 'version': 1}

                return {'success': True, 'data': beliefs_data, 'request_id': request_id}
            except Exception as e:
                return {'error': f'Failed to read beliefs: {e}', 'request_id': request_id}

        elif operation in ['write', 'update']:
            if not data:
                return {'error': 'Data required for write operation', 'request_id': request_id}

            try:
                # Load existing data
                if beliefs_path.exists():
                    with open(beliefs_path, 'r') as f:
                        beliefs_data = json.load(f)
                else:
                    beliefs_data = {'data': {}, 'version': 1}

                # Update data
                if operation == 'write':
                    beliefs_data['data'] = data
                else:  # update
                    beliefs_data['data'].update(data)

                beliefs_data['last_updated'] = datetime.utcnow().isoformat()
                beliefs_data['version'] = beliefs_data.get('version', 1) + 1
                beliefs_data['updated_by'] = request_id

                # Save data
                beliefs_path.parent.mkdir(parents=True, exist_ok=True)
                with open(beliefs_path, 'w') as f:
                    json.dump(beliefs_data, f, indent=2)

                return {'success': True, 'message': 'Beliefs updated', 'request_id': request_id}
            except Exception as e:
                return {'error': f'Failed to update beliefs: {e}', 'request_id': request_id}

        return {'error': f'Unsupported operation: {operation}', 'request_id': request_id}

    def _handle_ephemeral_request(self, operation: str, data: Optional[Dict[str, Any]],
                                 token: Optional[str], request_id: str) -> Dict[str, Any]:
        """Handle ephemeral layer requests (all authenticated users)."""
        token_type = get_token_type(token) if token else None

        if not token_type:
            return {'error': 'Authentication required for ephemeral layer', 'request_id': request_id}

        # Load ephemeral data
        ephemeral_path = Path("data/ephemeral_layer.json")

        if operation == 'read':
            try:
                if ephemeral_path.exists():
                    with open(ephemeral_path, 'r') as f:
                        ephemeral_data = json.load(f)
                else:
                    ephemeral_data = {'data': {}, 'version': 1}

                return {'success': True, 'data': ephemeral_data, 'request_id': request_id}
            except Exception as e:
                return {'error': f'Failed to read ephemeral: {e}', 'request_id': request_id}

        elif operation in ['write', 'update']:
            if not data:
                return {'error': 'Data required for write operation', 'request_id': request_id}

            try:
                # Load existing data
                if ephemeral_path.exists():
                    with open(ephemeral_path, 'r') as f:
                        ephemeral_data = json.load(f)
                else:
                    ephemeral_data = {'data': {}, 'version': 1}

                # Update data
                if operation == 'write':
                    ephemeral_data['data'] = data
                else:  # update
                    ephemeral_data['data'].update(data)

                ephemeral_data['last_updated'] = datetime.utcnow().isoformat()
                ephemeral_data['version'] = ephemeral_data.get('version', 1) + 1
                ephemeral_data['updated_by'] = request_id

                # Save data
                ephemeral_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ephemeral_path, 'w') as f:
                    json.dump(ephemeral_data, f, indent=2)

                return {'success': True, 'message': 'Ephemeral data updated', 'request_id': request_id}
            except Exception as e:
                return {'error': f'Failed to update ephemeral: {e}', 'request_id': request_id}

        return {'error': f'Unsupported operation: {operation}', 'request_id': request_id}

    def get_system_status(self) -> Dict[str, Any]:
        """Get HRM system status."""
        return {
            'status': 'healthy' if not self.config.get('safe_mode') else 'safe_mode',
            'timestamp': datetime.utcnow().isoformat(),
            'config': {
                'rate_limit_window': self.config['rate_limit_window'],
                'rate_limit_max': self.config['rate_limit_max'],
                'safe_mode': self.config['safe_mode'],
                'debug': self.config['debug'],
            },
            'layers': ['identity', 'beliefs', 'ephemeral'],
        }
