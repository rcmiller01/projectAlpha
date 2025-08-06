"""
Core Conductor - Enhanced with GraphRAG Memory and Tool Routing

This module demonstrates integration of the Core Conductor with the new
GraphRAG memory system and tool router for enhanced reasoning capabilities.

Key Features:
- Strategic reasoning enhanced with semantic memory
- Autonomous tool usage for expanded capabilities  
- Seamless integration with existing HRM stack
- Preparation for future SLiM agent integration

Author: ProjectAlpha Team
Compatible with: HRM stack, GraphRAG memory, Tool router
"""

import threading
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import the HRM Router integration
try:
    from .hrm_router import HRMRouter
except ImportError:
    from hrm_router import HRMRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConductorDecision:
    """Structured decision output from conductor"""
    decision_id: str
    strategic_context: str
    action_plan: List[str]
    memory_context: List[Dict[str, Any]]
    tool_recommendations: List[str]
    confidence: float
    reasoning_path: List[str]
    timestamp: str

class CoreConductor:
    """
    Enhanced Core Conductor with GraphRAG Memory and Tool Routing
    
    The conductor provides strategic, high-level reasoning and planning
    enhanced with semantic memory and autonomous tool capabilities.
    
    Features:
    - Strategic decision making with memory context
    - Autonomous tool discovery and usage
    - Integration with GraphRAG semantic memory
    - Preparation for multi-agent orchestration
    - Thread-safe concurrent operation
    """
    
    def __init__(self, 
                 memory_file: Optional[str] = None,
                 tool_log_file: Optional[str] = None,
                 conductor_id: Optional[str] = None):
        """
        Initialize Enhanced Core Conductor.
        
        Args:
            memory_file: Path to GraphRAG memory file
            tool_log_file: Path to tool request log
            conductor_id: Unique identifier for this conductor instance
        """
        self.conductor_id = conductor_id or f"conductor_{str(uuid.uuid4())[:8]}"
        self.hrm_router = HRMRouter(memory_file, tool_log_file)
        self._lock = threading.Lock()
        
        # Strategic context
        self.current_objectives: List[str] = []
        self.active_contexts: Dict[str, Any] = {}
        
        # Register conductor-specific tools
        self._register_conductor_tools()
        
        logger.info(f"Core Conductor {self.conductor_id} initialized with GraphRAG integration")
    
    def _register_conductor_tools(self):
        """Register tools specific to conductor-level operations"""
        
        def strategic_analysis_tool(context: str, objectives: Optional[List[str]] = None, **kwargs) -> dict:
            """Tool for strategic analysis and planning"""
            try:
                # Query memory for strategic context
                memory_response = self.hrm_router.execute_tool(
                    "query_memory", 
                    {"concept": "strategy", "depth": 3},
                    agent_type="conductor"
                )
                
                strategic_concepts = []
                if memory_response.success and memory_response.result:
                    strategic_concepts = memory_response.result.get('related_concepts', [])
                
                return {
                    "success": True,
                    "analysis": f"Strategic analysis for: {context}",
                    "context": context,
                    "objectives": objectives or [],
                    "strategic_memory": strategic_concepts[:5],  # Top 5 strategic concepts
                    "recommendations": [
                        "Evaluate current objectives alignment",
                        "Consider memory context for decisions",
                        "Identify tool requirements for execution"
                    ],
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
        
        def objective_planning_tool(goal: str, constraints: Optional[List[str]] = None, **kwargs) -> dict:
            """Tool for objective-based planning"""
            try:
                # Query memory for related planning concepts
                memory_response = self.hrm_router.execute_tool(
                    "query_memory",
                    {"concept": goal, "depth": 2},
                    agent_type="conductor"
                )
                
                related_concepts = []
                if memory_response.success and memory_response.result:
                    related_concepts = memory_response.result.get('related_concepts', [])
                
                # Generate action plan
                action_plan = [
                    f"Define success criteria for: {goal}",
                    "Gather relevant memory context",
                    "Identify required tools and resources",
                    "Create execution timeline",
                    "Monitor progress and adapt"
                ]
                
                return {
                    "success": True,
                    "goal": goal,
                    "constraints": constraints or [],
                    "action_plan": action_plan,
                    "memory_context": related_concepts[:3],
                    "estimated_complexity": "medium",
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
        
        # Register conductor tools
        self.hrm_router.tool_router.register_tool("strategic_analysis", strategic_analysis_tool)
        self.hrm_router.tool_router.register_tool("objective_planning", objective_planning_tool)
    
    def make_strategic_decision(self, 
                              situation: str, 
                              objectives: Optional[List[str]] = None,
                              constraints: Optional[List[str]] = None) -> ConductorDecision:
        """
        Make a strategic decision with full memory and tool integration.
        
        Args:
            situation: Description of the current situation
            objectives: List of current objectives
            constraints: Any constraints to consider
            
        Returns:
            ConductorDecision with comprehensive reasoning and recommendations
        """
        decision_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            logger.info(f"Conductor making strategic decision: {decision_id}")
            
            # Process input through HRM Router for memory context
            input_context = self.hrm_router.process_agent_input(
                situation, 
                agent_type="conductor",
                context={
                    'objectives': objectives or [],
                    'constraints': constraints or [],
                    'conductor_id': self.conductor_id
                }
            )
            
            reasoning_path = ["Initial situation assessment"]
            
            # Perform strategic analysis
            strategic_response = self.hrm_router.execute_tool(
                "strategic_analysis",
                {
                    "context": situation,
                    "objectives": objectives or []
                },
                agent_type="conductor"
            )
            
            if strategic_response.success:
                reasoning_path.append("Strategic analysis completed")
            
            # Create objective planning if objectives provided
            planning_recommendations = []
            if objectives:
                for objective in objectives:
                    planning_response = self.hrm_router.execute_tool(
                        "objective_planning",
                        {"goal": objective, "constraints": constraints},
                        agent_type="conductor"
                    )
                    if planning_response.success and planning_response.result:
                        planning_recommendations.extend(
                            planning_response.result.get('action_plan', [])
                        )
                
                reasoning_path.append("Objective planning completed")
            
            # Generate comprehensive action plan
            action_plan = []
            
            # Add strategic recommendations
            if strategic_response.success and strategic_response.result:
                action_plan.extend(strategic_response.result.get('recommendations', []))
            
            # Add planning recommendations
            action_plan.extend(planning_recommendations)
            
            # Add memory-informed actions
            memory_related = input_context.get('memory_related', [])
            if memory_related:
                action_plan.append("Leverage relevant memory context for execution")
                reasoning_path.append("Memory context integrated")
            
            # Add tool recommendations
            suggested_tools = input_context.get('suggested_tools', [])
            if suggested_tools:
                action_plan.append(f"Consider using tools: {', '.join(suggested_tools)}")
                reasoning_path.append("Tool recommendations identified")
            
            # Calculate confidence based on available context
            confidence = 0.7  # Base confidence
            if memory_related:
                confidence += 0.1  # Boost for memory context
            if strategic_response.success:
                confidence += 0.1  # Boost for successful strategic analysis
            if objectives:
                confidence += 0.1  # Boost for clear objectives
            
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            # Create decision object
            decision = ConductorDecision(
                decision_id=decision_id,
                strategic_context=situation,
                action_plan=action_plan,
                memory_context=memory_related,
                tool_recommendations=suggested_tools,
                confidence=confidence,
                reasoning_path=reasoning_path,
                timestamp=start_time.isoformat()
            )
            
            # Store decision in memory for future reference
            self._store_decision_in_memory(decision)
            
            # Process output through HRM Router
            output_summary = f"Strategic decision made with {len(action_plan)} recommended actions"
            self.hrm_router.process_agent_output(output_summary, input_context, "conductor")
            
            logger.info(f"Strategic decision completed: {decision_id} (confidence: {confidence:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making strategic decision: {e}")
            # Return error decision
            return ConductorDecision(
                decision_id=decision_id,
                strategic_context=situation,
                action_plan=[f"Error in decision making: {str(e)}"],
                memory_context=[],
                tool_recommendations=[],
                confidence=0.0,
                reasoning_path=["Error encountered"],
                timestamp=start_time.isoformat()
            )
    
    def _store_decision_in_memory(self, decision: ConductorDecision):
        """Store strategic decision in GraphRAG memory"""
        try:
            # Store decision as memory facts
            self.hrm_router.execute_tool(
                "add_memory_fact",
                {
                    "subject": "conductor",
                    "relation": "made_decision",
                    "object_node": decision.decision_id,
                    "confidence": decision.confidence,
                    "source": self.conductor_id
                },
                agent_type="conductor"
            )
            
            # Store action relationships
            for i, action in enumerate(decision.action_plan[:3]):  # Store top 3 actions
                self.hrm_router.execute_tool(
                    "add_memory_fact",
                    {
                        "subject": decision.decision_id,
                        "relation": "includes_action",
                        "object_node": action[:50],  # Truncate for brevity
                        "confidence": 0.9,
                        "source": self.conductor_id
                    },
                    agent_type="conductor"
                )
            
        except Exception as e:
            logger.error(f"Error storing decision in memory: {e}")
    
    def set_objectives(self, objectives: List[str]):
        """Set current strategic objectives"""
        with self._lock:
            self.current_objectives = objectives.copy()
            logger.info(f"Conductor objectives updated: {len(objectives)} objectives set")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current conductor status"""
        with self._lock:
            return {
                'conductor_id': self.conductor_id,
                'current_objectives': self.current_objectives.copy(),
                'active_contexts': len(self.active_contexts),
                'integration_stats': self.hrm_router.get_integration_stats()
            }
    
    def save_state(self) -> bool:
        """Save conductor state and memory"""
        return self.hrm_router.save_all()


def example_conductor_usage():
    """Example usage of Enhanced Core Conductor"""
    
    # Initialize conductor
    conductor = CoreConductor(
        memory_file="data/conductor_memory.json",
        tool_log_file="logs/conductor_tools.jsonl"
    )
    
    print("Enhanced Core Conductor Example")
    print("=" * 40)
    
    # Set strategic objectives
    objectives = [
        "Improve user engagement with AI companion",
        "Enhance emotional intelligence capabilities", 
        "Optimize response relevance and personalization"
    ]
    conductor.set_objectives(objectives)
    print(f"\nObjectives set: {len(objectives)} strategic goals")
    
    # Make strategic decision
    print("\n1. Strategic Decision Making:")
    situation = "User has been asking more complex emotional questions recently and seems to want deeper conversations"
    
    decision = conductor.make_strategic_decision(
        situation=situation,
        objectives=objectives,
        constraints=["Must maintain user privacy", "Stay within system capabilities"]
    )
    
    print(f"  Decision ID: {decision.decision_id}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Action Plan: {len(decision.action_plan)} actions")
    print(f"  Memory Context: {len(decision.memory_context)} related concepts")
    print(f"  Tool Recommendations: {decision.tool_recommendations}")
    
    # Show reasoning path
    print(f"\n2. Reasoning Path:")
    for i, step in enumerate(decision.reasoning_path, 1):
        print(f"  {i}. {step}")
    
    # Show action plan
    print(f"\n3. Action Plan:")
    for i, action in enumerate(decision.action_plan, 1):
        print(f"  {i}. {action}")
    
    # Get conductor status
    print(f"\n4. Conductor Status:")
    status = conductor.get_status()
    print(f"  ID: {status['conductor_id']}")
    print(f"  Objectives: {len(status['current_objectives'])}")
    print(f"  Memory Nodes: {status['integration_stats']['memory_stats']['total_nodes']}")
    print(f"  Available Tools: {status['integration_stats']['tool_stats']['total_tools']}")
    
    # Save state
    conductor.save_state()
    print(f"\n5. State saved successfully!")


if __name__ == "__main__":
    example_conductor_usage()
