#!/usr/bin/env python3
"""
Agent Bridge Python Service - Interface between Node.js and SLiM agents

This service provides a JSON-based communication bridge between the Node.js
backend and the ProjectAlpha SLiM agent system.

Communication Protocol:
- Input: JSON messages via stdin
- Output: JSON responses via stdout
- Error handling: Structured error messages

Message Types:
- init: Initialize the bridge
- invoke_agent: Call a specific SLiM agent
- get_agents: List available agents
- shutdown: Clean shutdown

Author: ProjectAlpha Team
"""

import sys
import json
import logging
import asyncio
import re
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.hrm_router import HRMRouter
    from src.core.core_conductor import CoreConductor
    from memory.graphrag_memory import GraphRAGMemory
    from src.tools.tool_request_router import ToolRequestRouter
    from src.agents.deduction_agent import DeductionAgent
    from src.agents.metaphor_agent import MetaphorAgent
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)

# Configure logging to stderr to avoid interfering with JSON communication
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class AgentBridgeService:
    """
    Python service for bridging Node.js requests to SLiM agents with input validation and security
    """

    def __init__(self):
        self.hrm_router = None
        self.initialized = False
        self.agents_cache = {}

        # Security configuration
        self.require_anchor_confirmation = True
        self.anchor_confirm_token = os.getenv("ANCHOR_CONFIRM_TOKEN", "anchor_confirmed")
        self.max_prompt_length = 2000
        self.max_context_items = 50

    def validate_input(self, data: dict) -> tuple[bool, str]:
        """
        Validate and sanitize input data for security.

        Args:
            data: Input data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for required fields based on message type
            msg_type = data.get('type')

            if not msg_type:
                return False, "Missing message type"

            if msg_type not in ['init', 'invoke_agent', 'get_agents', 'shutdown']:
                return False, f"Unknown message type: {msg_type}"

            # Validate payload if present
            payload = data.get('payload', {})

            if msg_type == 'invoke_agent':
                # Validate agent invocation parameters
                agent_type = payload.get('agent_type')
                prompt = payload.get('prompt')

                if not agent_type:
                    return False, "Missing agent_type in payload"

                if not prompt:
                    return False, "Missing prompt in payload"

                # Validate prompt length
                if len(prompt) > self.max_prompt_length:
                    return False, f"Prompt too long (max {self.max_prompt_length} characters)"

                # Check for dangerous patterns
                dangerous_patterns = [
                    r'<script[^>]*>.*?</script>',  # XSS
                    r'javascript:',               # JavaScript protocol
                    r'on\w+\s*=',                # Event handlers
                    r'eval\s*\(',                # Code injection
                    r'exec\s*\(',                # Code execution
                ]

                for pattern in dangerous_patterns:
                    if re.search(pattern, prompt, re.IGNORECASE):
                        return False, f"Prompt contains potentially dangerous content"

                # Validate context if provided
                context = payload.get('context', {})
                if context and len(context) > self.max_context_items:
                    return False, f"Too many context items (max {self.max_context_items})"

                # Validate depth parameter
                depth = payload.get('depth', 1)
                if not isinstance(depth, int) or depth < 1 or depth > 5:
                    return False, "Depth must be an integer between 1 and 5"

            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def sanitize_input(self, data: dict) -> dict:
        """
        Sanitize input data by removing/cleaning dangerous content.

        Args:
            data: Input data to sanitize

        Returns:
            Sanitized data dictionary
        """
        try:
            sanitized = data.copy()

            if 'payload' in sanitized:
                payload = sanitized['payload']

                # Sanitize prompt if present
                if 'prompt' in payload:
                    prompt = str(payload['prompt'])
                    # Remove HTML tags and suspicious patterns
                    prompt = re.sub(r'<[^>]+>', '', prompt)
                    prompt = re.sub(r'javascript:', '', prompt, flags=re.IGNORECASE)
                    prompt = re.sub(r'on\w+\s*=', '', prompt, flags=re.IGNORECASE)

                    # Limit length
                    if len(prompt) > self.max_prompt_length:
                        prompt = prompt[:self.max_prompt_length] + "... [truncated]"

                    payload['prompt'] = prompt

                # Sanitize context items
                if 'context' in payload and isinstance(payload['context'], dict):
                    context = payload['context']
                    sanitized_context = {}

                    for key, value in list(context.items())[:self.max_context_items]:
                        # Clean key and value
                        clean_key = re.sub(r'[<>"\']', '', str(key))
                        clean_value = re.sub(r'[<>"\']', '', str(value))
                        sanitized_context[clean_key] = clean_value

                    payload['context'] = sanitized_context

            return sanitized

        except Exception as e:
            logger.error(f"Error sanitizing input: {e}")
            return data

    def check_anchor_confirmation(self, data: dict) -> tuple[bool, str]:
        """
        Check for anchor confirmation when required for sensitive operations.

        Args:
            data: Input data to check

        Returns:
            Tuple of (confirmed, error_message)
        """
        if not self.require_anchor_confirmation:
            return True, ""

        msg_type = data.get('type')

        # Operations requiring anchor confirmation
        sensitive_operations = ['invoke_agent']

        if msg_type in sensitive_operations:
            anchor_confirm = data.get('anchor_confirm')

            if not anchor_confirm:
                return False, "Anchor confirmation required for agent forwarding"

            if anchor_confirm != self.anchor_confirm_token:
                logger.warning(f"Invalid anchor confirmation token provided: {anchor_confirm[:8]}...")
                return False, "Invalid anchor confirmation token"

            logger.info(f"Anchor confirmation verified for {msg_type}")

        return True, ""

    async def initialize(self, project_root_path):
        """Initialize the SLiM agent system"""
        try:
            if not IMPORTS_SUCCESSFUL:
                raise RuntimeError(f"Failed to import SLiM components: {IMPORT_ERROR}")

            logger.info("Initializing SLiM agent system...")

            # Initialize HRM Router with GraphRAG and tool integration
            self.hrm_router = HRMRouter()

            logger.info("SLiM agent system initialized successfully")
            self.initialized = True

            # Cache available agents
            self.agents_cache = self.hrm_router.list_agents()

            return {
                'success': True,
                'message': 'SLiM agent system initialized',
                'agents_count': len(self.agents_cache)
            }

        except Exception as e:
            logger.error(f"Failed to initialize SLiM agent system: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def invoke_agent(self, agent_type, prompt, depth=1, context=None):
        """Invoke a specific SLiM agent"""
        try:
            if not self.initialized:
                raise RuntimeError("Agent system not initialized")

            logger.info(f"Invoking {agent_type} agent with prompt: {prompt[:100]}...")

            # Use HRM router to dispatch to agent
            def dispatch_agent():
                kwargs = {"depth": depth}
                if context:
                    kwargs.update(context)
                return self.hrm_router.dispatch_to_agent(agent_type, prompt, **kwargs)

            response = await asyncio.get_event_loop().run_in_executor(None, dispatch_agent)

            return {
                'success': True,
                'agent_type': agent_type,
                'response': response,
                'timestamp': str(asyncio.get_event_loop().time())
            }

        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_type': agent_type
            }

    async def get_agents(self):
        """Get list of available agents"""
        try:
            if not self.initialized:
                return {
                    'success': False,
                    'error': 'Agent system not initialized'
                }

            agents = self.hrm_router.list_agents()

            return {
                'success': True,
                'agents': agents,
                'count': len(agents)
            }

        except Exception as e:
            logger.error(f"Failed to get agents: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def send_response(self, response_type, request_id=None, payload=None, error=None):
        """Send JSON response to Node.js"""
        message = {
            'type': response_type,
            'request_id': request_id,
            'payload': payload,
            'error': error,
            'timestamp': str(asyncio.get_event_loop().time())
        }

        print(json.dumps(message), flush=True)

    async def handle_message(self, message):
        """Handle incoming message from Node.js with validation and security checks"""
        try:
            # Validate input
            is_valid, validation_error = self.validate_input(message)
            if not is_valid:
                logger.warning(f"Input validation failed: {validation_error}")
                self.send_response('error', message.get('request_id'), None, f"Validation error: {validation_error}")
                return

            # Check anchor confirmation for sensitive operations
            confirmed, confirm_error = self.check_anchor_confirmation(message)
            if not confirmed:
                logger.warning(f"Anchor confirmation failed: {confirm_error}")
                self.send_response('error', message.get('request_id'), None, f"Security error: {confirm_error}")
                return

            # Sanitize input
            sanitized_message = self.sanitize_input(message)

            # Process sanitized message
            msg_type = sanitized_message.get('type')
            request_id = sanitized_message.get('request_id')
            payload = sanitized_message.get('payload', {})

            logger.info(f"Processing validated message: {msg_type}")

            if msg_type == 'init':
                project_root = payload.get('project_root')
                result = await self.initialize(project_root)
                self.send_response('init_response', request_id, result)

            elif msg_type == 'invoke_agent':
                agent_type = payload.get('agent_type')
                prompt = payload.get('prompt')
                depth = payload.get('depth', 1)
                context = payload.get('context', {})

                logger.info(f"Forwarding validated request to agent {agent_type} (anchor confirmed)")
                result = await self.invoke_agent(agent_type, prompt, depth, context)
                self.send_response('agent_response', request_id, result)

            elif msg_type == 'get_agents':
                result = await self.get_agents()
                self.send_response('agent_list', request_id, result)

            elif msg_type == 'shutdown':
                logger.info("Shutdown requested")
                sys.exit(0)

            else:
                self.send_response('error', request_id, None, f"Unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.send_response('error', message.get('request_id'), None, str(e))

async def main():
    """Main event loop for the agent bridge service"""
    logger.info("🐍 Starting Agent Bridge Python Service...")

    bridge = AgentBridgeService()

    # Send startup message
    bridge.send_response('startup', None, {
        'message': 'Agent Bridge Python Service started',
        'imports_successful': IMPORTS_SUCCESSFUL
    })

    # Main message processing loop
    try:
        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

            if not line:  # EOF
                break

            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
                await bridge.handle_message(message)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                bridge.send_response('error', None, None, f"Invalid JSON: {str(e)}")

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        logger.info("Agent Bridge Python Service shutdown")

if __name__ == '__main__':
    asyncio.run(main())
