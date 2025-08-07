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
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core.hrm_router import HRMRouter
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
    Python service for bridging Node.js requests to SLiM agents
    """
    
    def __init__(self):
        self.hrm_router = None
        self.initialized = False
        self.agents_cache = {}
        
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
        """Handle incoming message from Node.js"""
        try:
            msg_type = message.get('type')
            request_id = message.get('request_id')
            payload = message.get('payload', {})
            
            if msg_type == 'init':
                project_root = payload.get('project_root')
                result = await self.initialize(project_root)
                self.send_response('init_response', request_id, result)
                
            elif msg_type == 'invoke_agent':
                agent_type = payload.get('agent_type')
                prompt = payload.get('prompt')
                depth = payload.get('depth', 1)
                context = payload.get('context', {})
                
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
            self.send_response('error', request_id, None, str(e))

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
