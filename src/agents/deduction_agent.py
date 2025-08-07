"""
DeductionAgent - Left-Brain Logic Specialist

This module implements a specialized SLiM agent focused on logical reasoning,
mathematical deduction, and analytical problem solving.

The DeductionAgent leverages the logic_high model for advanced reasoning tasks
including formal logic, mathematical proofs, analytical thinking, and 
evidence-based conclusions.

Capabilities:
- Mathematical reasoning and proofs
- Formal logic analysis
- Deductive reasoning chains
- Evidence synthesis and validation
- Analytical problem decomposition

Author: ProjectAlpha Team
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import SLiMAgent

logger = logging.getLogger(__name__)

class DeductionAgent(SLiMAgent):
    """
    Specialized SLiM agent for logical reasoning and mathematical deduction.
    
    This agent uses the logic_high model to provide:
    - Advanced mathematical reasoning
    - Formal logic operations
    - Deductive reasoning chains
    - Evidence-based analysis
    - Analytical problem solving
    """
    
    def __init__(self, conductor, memory, router, agent_id=None):
        """
        Initialize DeductionAgent with logic_high model role.
        
        Args:
            conductor: CoreConductor instance for model access
            memory: GraphRAG memory system for semantic context
            router: Tool request router for autonomous capabilities
            agent_id: Optional custom agent identifier
        """
        super().__init__(
            role="logic_high", 
            conductor=conductor, 
            memory=memory, 
            router=router,
            agent_id=agent_id or "deduction_agent"
        )
        
        # DeductionAgent-specific configuration
        self.reasoning_style = "deductive"
        self.confidence_threshold = 0.8  # High confidence for logical reasoning
        self.max_proof_steps = 10
        
        logger.info(f"DeductionAgent {self.agent_id} initialized for logical reasoning")
    
    def prove(self, statement: str, premises: Optional[List[str]] = None) -> str:
        """
        Attempt to prove a logical statement given premises.
        
        Args:
            statement: The statement to prove
            premises: Optional list of premises to work from
            
        Returns:
            Proof attempt or logical analysis
        """
        prompt_parts = [f"Prove the following statement: {statement}"]
        
        if premises:
            prompt_parts.append("Given premises:")
            for i, premise in enumerate(premises, 1):
                prompt_parts.append(f"{i}. {premise}")
        
        prompt_parts.append("Provide a step-by-step logical proof or analysis.")
        
        full_prompt = "\n".join(prompt_parts)
        
        logger.debug(f"DeductionAgent proving: {statement}")
        return self.run(full_prompt, depth=2, use_tools=True)
    
    def analyze_argument(self, argument: str) -> str:
        """
        Analyze the logical validity of an argument.
        
        Args:
            argument: The argument to analyze
            
        Returns:
            Logical analysis of the argument's validity
        """
        prompt = f"""
        Analyze the logical structure and validity of this argument:
        
        {argument}
        
        Provide:
        1. Identification of premises and conclusion
        2. Logical form analysis
        3. Validity assessment
        4. Any logical fallacies identified
        """
        
        logger.debug(f"DeductionAgent analyzing argument validity")
        return self.run(prompt, depth=2, use_tools=True)
    
    def solve_mathematical(self, problem: str, show_work: bool = True) -> str:
        """
        Solve a mathematical problem with logical reasoning.
        
        Args:
            problem: Mathematical problem to solve
            show_work: Whether to show step-by-step work
            
        Returns:
            Mathematical solution with reasoning
        """
        prompt_parts = [f"Solve this mathematical problem: {problem}"]
        
        if show_work:
            prompt_parts.append("Show all steps and reasoning clearly.")
        
        full_prompt = "\n".join(prompt_parts)
        
        logger.debug(f"DeductionAgent solving mathematical problem")
        return self.run(full_prompt, depth=1, use_tools=True)
    
    def deduce_from_facts(self, facts: List[str], question: Optional[str] = None) -> str:
        """
        Perform deductive reasoning from a set of facts.
        
        Args:
            facts: List of known facts
            question: Optional specific question to answer
            
        Returns:
            Deductive conclusions from the facts
        """
        prompt_parts = ["Given these facts:"]
        
        for i, fact in enumerate(facts, 1):
            prompt_parts.append(f"{i}. {fact}")
        
        if question:
            prompt_parts.append(f"\nAnswer this question: {question}")
        else:
            prompt_parts.append("\nWhat logical conclusions can be drawn?")
        
        full_prompt = "\n".join(prompt_parts)
        
        logger.debug(f"DeductionAgent performing deductive reasoning")
        return self.run(full_prompt, depth=2, use_tools=True)
    
    def _evaluate_tool_usage(self, prompt: str, context: List[Dict[str, Any]]) -> List[str]:
        """
        Enhanced tool evaluation for logical reasoning tasks.
        """
        tool_calls = super()._evaluate_tool_usage(prompt, context)
        
        prompt_lower = prompt.lower()
        
        # Logic-specific tool usage patterns
        if any(word in prompt_lower for word in ["prove", "proof", "theorem"]):
            tool_calls.append("logical_proof_assistant")
        
        if any(word in prompt_lower for word in ["calculate", "solve", "compute"]):
            tool_calls.append("mathematical_calculator")
        
        if any(word in prompt_lower for word in ["validate", "verify", "check"]):
            tool_calls.append("logic_validator")
        
        return tool_calls
    
    def _update_memory(self, prompt: str, response: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced memory updates for logical reasoning.
        """
        memory_updates = super()._update_memory(prompt, response, context)
        
        try:
            # Extract logical facts and relationships
            prompt_lower = prompt.lower()
            
            # Store mathematical solutions
            if any(word in prompt_lower for word in ["solve", "calculate", "prove"]):
                fact = {
                    "subject": "mathematical_reasoning",
                    "predicate": "solved_problem",
                    "object": prompt[:150] + "..." if len(prompt) > 150 else prompt,
                    "metadata": {
                        "agent": self.agent_id,
                        "reasoning_type": "mathematical",
                        "solution_length": len(response)
                    }
                }
                
                self.memory.add_fact(
                    fact["subject"],
                    fact["predicate"],
                    fact["object"],
                    fact["metadata"]
                )
                memory_updates.append(fact)
            
            # Store logical proofs
            if any(word in prompt_lower for word in ["prove", "proof", "theorem"]):
                fact = {
                    "subject": "logical_proof",
                    "predicate": "attempted_proof",
                    "object": prompt[:150] + "..." if len(prompt) > 150 else prompt,
                    "metadata": {
                        "agent": self.agent_id,
                        "reasoning_type": "deductive",
                        "proof_complexity": "high" if len(response) > 500 else "medium"
                    }
                }
                
                self.memory.add_fact(
                    fact["subject"],
                    fact["predicate"],
                    fact["object"],
                    fact["metadata"]
                )
                memory_updates.append(fact)
                
        except Exception as e:
            logger.warning(f"Enhanced memory update failed: {e}")
        
        return memory_updates
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics specific to logical reasoning performance."""
        base_stats = self.get_status()
        
        reasoning_stats = {
            "reasoning_style": self.reasoning_style,
            "confidence_threshold": self.confidence_threshold,
            "max_proof_steps": self.max_proof_steps,
            "specialization": "logical_reasoning_and_mathematical_deduction"
        }
        
        return {**base_stats, **reasoning_stats}
    
    def get_valence(self, prompt: str) -> float:
        """
        Mock method to calculate valence score for a given prompt.

        Args:
            prompt: The input prompt to evaluate.

        Returns:
            A float representing the valence score.
        """
        logger.debug(f"Calculating valence for prompt: {prompt}")
        # Mock implementation: return a random score between -1 and 1
        import random
        return random.uniform(-1, 1)
