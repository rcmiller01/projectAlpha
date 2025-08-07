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
    from .init_models import load_conductor_models, ModelInterface
except ImportError:
    from hrm_router import HRMRouter
    from init_models import load_conductor_models, ModelInterface

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
                 conductor_id: Optional[str] = None,
                 load_models: bool = True):
        """
        Initialize Enhanced Core Conductor.
        
        Args:
            memory_file: Path to GraphRAG memory file
            tool_log_file: Path to tool request log
            conductor_id: Unique identifier for this conductor instance
            load_models: Whether to load AI models on initialization
        """
        self.conductor_id = conductor_id or f"conductor_{str(uuid.uuid4())[:8]}"
        self.hrm_router = HRMRouter(memory_file, tool_log_file)
        self._lock = threading.Lock()
        
        # Strategic context
        self.current_objectives: List[str] = []
        self.active_contexts: Dict[str, Any] = {}
        
        # Model management
        self.models: Dict[str, ModelInterface] = {}
        if load_models:
            self.init_models()
        
        # Register conductor-specific tools
        self._register_conductor_tools()
        
        logger.info(f"Core Conductor {self.conductor_id} initialized with GraphRAG integration")
    
    def init_models(self) -> None:
        """
        Initialize and load all AI models for the conductor.
        
        Automatically detects and loads either:
        - Standard conductor models (conductor, logic, emotion, creative)
        - Full SLiM agent suite (conductor + 8 SLiM agents) if configured
        
        SLiM Agent Roles:
        Left Brain (Logic): logic_high, logic_code, logic_proof, logic_fallback
        Right Brain (Emotion): emotion_valence, emotion_narrative, emotion_uncensored, emotion_creative
        """
        logger.info(f"Initializing models for Core Conductor {self.conductor_id}")
        
        try:
            # Check if SLiM environment variables are configured
            import os
            has_slim_config = any(os.getenv(var) for var in [
                "LOGIC_HIGH_MODEL", "LOGIC_CODE_MODEL", "LOGIC_PROOF_MODEL", "LOGIC_FALLBACK_MODEL",
                "EMOTION_VALENCE_MODEL", "EMOTION_NARRATIVE_MODEL", "EMOTION_UNCENSORED_MODEL", "EMOTION_CREATIVE_MODEL"
            ])
            
            if has_slim_config:
                from .init_models import load_all_models
                self.models = load_all_models()
                logger.info("Loaded SLiM agent model suite (Conductor + 8 SLiMs + legacy roles)")
            else:
                self.models = load_conductor_models()
                logger.info("Loaded standard conductor model suite")
            
            # Log loaded models
            for role, model in self.models.items():
                model_type = type(model).__name__
                model_name = getattr(model, 'model_name', 'unknown')
                logger.info(f"  {role}: {model_type} ({model_name})")
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fallback to empty dict to prevent crashes
            self.models = {}
            raise
            # Ensure we have at least mock models
            from .init_models import MockModel
            self.models = {
                "conductor": MockModel("fallback-conductor"),
                "logic": MockModel("fallback-logic"),
                "emotion": MockModel("fallback-emotion"),
                "creative": MockModel("fallback-creative")
            }
            logger.warning("Loaded fallback mock models due to initialization error")
    
    def generate(self, role: str, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response using the specified model role.
        
        Args:
            role: Model role to use. Standard roles: "conductor", "logic", "emotion", "creative"
                  SLiM agent roles: "logic_high", "logic_code", "logic_proof", "logic_fallback",
                                   "emotion_valence", "emotion_narrative", "emotion_uncensored", "emotion_creative"
            prompt: The prompt to send to the model
            context: Optional context to include with the prompt
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Generated response string
            
        Raises:
            ValueError: If the specified role is not available
        """
        if role not in self.models:
            available_roles = list(self.models.keys())
            raise ValueError(f"Role '{role}' not available. Available roles: {available_roles}")
        
        try:
            # Get the model for this role
            model = self.models[role]
            
            # Generate response
            response = model.generate(prompt, context=context, **kwargs)
            
            logger.debug(f"Generated response using {role} model: {len(response)} characters")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with {role} model: {e}")
            # Fallback response
            return f"[ERROR: Failed to generate response with {role} model: {str(e)}]"
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about loaded models"""
        from .init_models import get_model_info
        
        status = {
            "total_models": len(self.models),
            "loaded_roles": list(self.models.keys()),
            "models": {}
        }
        
        for role, model in self.models.items():
            status["models"][role] = get_model_info(model)
        
        return status
    
    def reload_model(self, role: str) -> bool:
        """
        Reload a specific model role.
        
        Args:
            role: Model role to reload
            
        Returns:
            Success status
        """
        try:
            from .init_models import load_model
            
            # Default model mappings for each role
            role_defaults = {
                "conductor": "llama3.1:8b",
                "logic": "deepseek-coder:1.3b", 
                "emotion": "mistral:7b",
                "creative": "mixtral:8x7b"
            }
            
            if role not in role_defaults:
                logger.error(f"Unknown role '{role}' for model reload")
                return False
            
            # Load the model
            env_var = f"{role.upper()}_MODEL"
            default_model = role_defaults[role]
            
            new_model = load_model(env_var, default_model)
            
            # Replace the model
            with self._lock:
                self.models[role] = new_model
            
            logger.info(f"Successfully reloaded {role} model")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading {role} model: {e}")
            return False
    
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
            
            # Enhanced reasoning with AI models if available
            if self.models:
                try:
                    # Use conductor model for strategic synthesis
                    synthesis_prompt = f"""
                    Situation: {situation}
                    Objectives: {objectives or 'None specified'}
                    Constraints: {constraints or 'None specified'}
                    Memory Context: {len(memory_related)} related concepts found
                    
                    Provide strategic synthesis and key insights for this decision.
                    """
                    
                    strategic_synthesis = self.generate("conductor", synthesis_prompt, context=str(memory_related[:3]))
                    if strategic_synthesis and not strategic_synthesis.startswith("[ERROR"):
                        action_plan.append(f"Strategic insight: {strategic_synthesis[:100]}...")
                        reasoning_path.append("AI strategic synthesis completed")
                        confidence += 0.05  # Small boost for AI insights
                    
                    # Use logic model for constraint analysis if constraints exist
                    if constraints:
                        logic_prompt = f"Analyze the logical implications and potential conflicts of these constraints: {constraints}"
                        logic_analysis = self.generate("logic", logic_prompt)
                        if logic_analysis and not logic_analysis.startswith("[ERROR"):
                            action_plan.append(f"Constraint analysis: {logic_analysis[:100]}...")
                            reasoning_path.append("Logical constraint analysis completed")
                    
                except Exception as e:
                    logger.warning(f"Error in AI model reasoning: {e}")
                    reasoning_path.append("AI reasoning attempted but failed")
            
            confidence = min(confidence, 1.0)  # Re-cap after AI boost
            
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
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list-models":
        # List all loaded models
        print("Loading CoreConductor models...")
        conductor = CoreConductor()
        
        print(f"\n✅ Successfully loaded {len(conductor.models)} models:")
        print("-" * 60)
        
        # Group models by type
        standard_models = {}
        slim_models = {}
        
        for role, model in conductor.models.items():
            model_type = type(model).__name__
            model_name = getattr(model, 'model_name', 'unknown')
            
            if role in ["conductor", "logic", "emotion", "creative"]:
                standard_models[role] = (model_type, model_name)
            else:
                slim_models[role] = (model_type, model_name)
        
        # Display standard models
        if standard_models:
            print("📋 Standard Models:")
            for role, (model_type, model_name) in standard_models.items():
                print(f"  {role:<12}: {model_type:<12} ({model_name})")
        
        # Display SLiM models  
        if slim_models:
            print("\n🧠 SLiM Agent Models:")
            
            # Left brain (logic) models
            left_brain = {k: v for k, v in slim_models.items() if k.startswith('logic_')}
            if left_brain:
                print("  🧮 Left Brain (Logic):")
                for role, (model_type, model_name) in left_brain.items():
                    print(f"    {role:<16}: {model_type:<12} ({model_name})")
            
            # Right brain (emotion/creativity) models
            right_brain = {k: v for k, v in slim_models.items() if k.startswith('emotion_')}
            if right_brain:
                print("  🎨 Right Brain (Emotion/Creativity):")
                for role, (model_type, model_name) in right_brain.items():
                    print(f"    {role:<16}: {model_type:<12} ({model_name})")
        
        print("-" * 60)
        print(f"Total models loaded: {len(conductor.models)}")
        
    else:
        example_conductor_usage()
