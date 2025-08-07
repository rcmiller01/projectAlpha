"""
MetaphorAgent - Right-Brain Creativity Specialist

This module implements a specialized SLiM agent focused on creative expression,
metaphorical thinking, and artistic communication.

The MetaphorAgent leverages the emotion_creative model for creative tasks
including metaphor generation, artistic expression, creative writing, and
innovative conceptual connections.

Capabilities:
- Metaphor and analogy generation
- Creative writing and storytelling
- Artistic expression and interpretation
- Innovative conceptual connections
- Emotional and symbolic communication

Author: ProjectAlpha Team
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import SLiMAgent

logger = logging.getLogger(__name__)

class MetaphorAgent(SLiMAgent):
    """
    Specialized SLiM agent for creative expression and metaphorical thinking.
    
    This agent uses the emotion_creative model to provide:
    - Metaphor and analogy generation
    - Creative writing and storytelling
    - Artistic interpretation
    - Innovative conceptual connections
    - Emotional and symbolic expression
    """
    
    def __init__(self, conductor, memory, router, agent_id=None):
        """
        Initialize MetaphorAgent with emotion_creative model role.
        
        Args:
            conductor: CoreConductor instance for model access
            memory: GraphRAG memory system for semantic context
            router: Tool request router for autonomous capabilities
            agent_id: Optional custom agent identifier
        """
        super().__init__(
            role="emotion_creative", 
            conductor=conductor, 
            memory=memory, 
            router=router,
            agent_id=agent_id or "metaphor_agent"
        )
        
        # MetaphorAgent-specific configuration
        self.creativity_style = "metaphorical"
        self.expression_depth = "symbolic"
        self.innovation_threshold = 0.7  # Encourage creative thinking
        self.metaphor_complexity = "rich"
        
        logger.info(f"MetaphorAgent {self.agent_id} initialized for creative expression")
    
    def generate_metaphors(self, concept: str, count: int = 3, style: str = "rich") -> str:
        """
        Generate metaphors for a given concept.
        
        Args:
            concept: The concept to create metaphors for
            count: Number of metaphors to generate
            style: Style of metaphors ("simple", "rich", "poetic", "visual")
            
        Returns:
            Generated metaphors with explanations
        """
        prompt = f"""
        Create {count} {style} metaphors for the concept: {concept}
        
        For each metaphor:
        1. Present the metaphor clearly
        2. Explain the connection and symbolism
        3. Explore the emotional or conceptual resonance
        
        Make each metaphor unique and evocative.
        """
        
        logger.debug(f"MetaphorAgent generating metaphors for: {concept}")
        return self.run(prompt, depth=2, use_tools=True)
    
    def creative_story(self, theme: str, elements: Optional[List[str]] = None, 
                      style: str = "narrative") -> str:
        """
        Create a creative story around a theme.
        
        Args:
            theme: Central theme for the story
            elements: Optional story elements to include
            style: Writing style ("narrative", "poetic", "abstract", "symbolic")
            
        Returns:
            Creative story with thematic depth
        """
        prompt_parts = [f"Write a {style} story exploring the theme: {theme}"]
        
        if elements:
            prompt_parts.append("Include these elements:")
            for element in elements:
                prompt_parts.append(f"- {element}")
        
        prompt_parts.extend([
            "",
            "Focus on:",
            "- Rich imagery and symbolism",
            "- Emotional depth and resonance", 
            "- Creative and unexpected connections",
            "- Evocative language and metaphor"
        ])
        
        full_prompt = "\n".join(prompt_parts)
        
        logger.debug(f"MetaphorAgent creating story with theme: {theme}")
        return self.run(full_prompt, depth=2, use_tools=True)
    
    def interpret_symbol(self, symbol: str, context: Optional[str] = None) -> str:
        """
        Provide creative interpretation of a symbol.
        
        Args:
            symbol: The symbol to interpret
            context: Optional context for interpretation
            
        Returns:
            Rich symbolic interpretation with multiple layers
        """
        prompt_parts = [f"Provide a creative and deep interpretation of this symbol: {symbol}"]
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        prompt_parts.extend([
            "",
            "Explore:",
            "- Multiple layers of meaning",
            "- Cultural and universal symbolism",
            "- Emotional and psychological associations",
            "- Creative and personal interpretations",
            "- Connections to broader themes and archetypes"
        ])
        
        full_prompt = "\n".join(prompt_parts)
        
        logger.debug(f"MetaphorAgent interpreting symbol: {symbol}")
        return self.run(full_prompt, depth=2, use_tools=True)
    
    def creative_connection(self, concept1: str, concept2: str, 
                           connection_type: str = "metaphorical") -> str:
        """
        Find creative connections between two concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            connection_type: Type of connection ("metaphorical", "symbolic", "narrative", "emotional")
            
        Returns:
            Creative exploration of connections between concepts
        """
        prompt = f"""
        Explore the {connection_type} connections between: {concept1} and {concept2}
        
        Create innovative bridges between these concepts through:
        - Metaphorical parallels
        - Symbolic resonances
        - Emotional correspondences
        - Narrative possibilities
        - Unexpected associations
        
        Be creative and explore non-obvious connections.
        """
        
        logger.debug(f"MetaphorAgent connecting: {concept1} <-> {concept2}")
        return self.run(prompt, depth=2, use_tools=True)
    
    def emotional_landscape(self, emotion: str, medium: str = "visual") -> str:
        """
        Create an artistic representation of an emotional landscape.
        
        Args:
            emotion: The emotion to explore
            medium: Artistic medium ("visual", "musical", "poetic", "abstract")
            
        Returns:
            Creative emotional landscape description
        """
        prompt = f"""
        Create a {medium} landscape that embodies the emotion: {emotion}
        
        Describe this emotional landscape through:
        - Sensory details and imagery
        - Symbolic elements and metaphors
        - Color, texture, movement, sound
        - Spatial relationships and dynamics
        - Atmospheric qualities and mood
        
        Make it vivid, evocative, and emotionally resonant.
        """
        
        logger.debug(f"MetaphorAgent creating emotional landscape for: {emotion}")
        return self.run(prompt, depth=1, use_tools=True)
    
    def innovative_analogy(self, problem: str, domain: str = "nature") -> str:
        """
        Create innovative analogies to understand a problem.
        
        Args:
            problem: The problem or concept to understand
            domain: Domain to draw analogies from ("nature", "music", "architecture", "cooking")
            
        Returns:
            Creative analogies that illuminate the problem
        """
        prompt = f"""
        Create innovative analogies from the domain of {domain} to illuminate this problem or concept:
        {problem}
        
        For each analogy:
        1. Present the {domain}-based analogy clearly
        2. Explain how it maps to the original problem
        3. Explore what insights this reveals
        4. Suggest creative solutions or perspectives
        
        Be inventive and find unexpected parallels.
        """
        
        logger.debug(f"MetaphorAgent creating analogies for: {problem}")
        return self.run(prompt, depth=2, use_tools=True)
    
    def _evaluate_tool_usage(self, prompt: str, context: List[Dict[str, Any]]) -> List[str]:
        """
        Enhanced tool evaluation for creative tasks.
        """
        tool_calls = super()._evaluate_tool_usage(prompt, context)
        
        prompt_lower = prompt.lower()
        
        # Creativity-specific tool usage patterns
        if any(word in prompt_lower for word in ["metaphor", "analogy", "symbol"]):
            tool_calls.append("symbolism_database")
        
        if any(word in prompt_lower for word in ["story", "narrative", "creative"]):
            tool_calls.append("creative_writing_assistant")
        
        if any(word in prompt_lower for word in ["emotion", "feeling", "mood"]):
            tool_calls.append("emotional_analysis")
        
        if any(word in prompt_lower for word in ["color", "visual", "artistic"]):
            tool_calls.append("artistic_palette")
        
        return tool_calls
    
    def _update_memory(self, prompt: str, response: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced memory updates for creative expression.
        """
        memory_updates = super()._update_memory(prompt, response, context)
        
        try:
            # Extract creative facts and relationships
            prompt_lower = prompt.lower()
            
            # Store metaphor creations
            if any(word in prompt_lower for word in ["metaphor", "analogy"]):
                fact = {
                    "subject": "metaphorical_thinking",
                    "predicate": "created_metaphor",
                    "object": prompt[:150] + "..." if len(prompt) > 150 else prompt,
                    "metadata": {
                        "agent": self.agent_id,
                        "creativity_type": "metaphorical",
                        "expression_depth": self.expression_depth,
                        "response_richness": "high" if len(response) > 400 else "medium"
                    }
                }
                
                self.memory.add_fact(
                    fact["subject"],
                    fact["predicate"],
                    fact["object"],
                    fact["metadata"]
                )
                memory_updates.append(fact)
            
            # Store creative works
            if any(word in prompt_lower for word in ["story", "creative", "artistic"]):
                fact = {
                    "subject": "creative_expression",
                    "predicate": "generated_work",
                    "object": prompt[:150] + "..." if len(prompt) > 150 else prompt,
                    "metadata": {
                        "agent": self.agent_id,
                        "creativity_type": "artistic",
                        "work_complexity": "complex" if len(response) > 600 else "moderate",
                        "innovation_level": self.innovation_threshold
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
    
    def get_creativity_stats(self) -> Dict[str, Any]:
        """Get statistics specific to creative performance."""
        base_stats = self.get_status()
        
        creativity_stats = {
            "creativity_style": self.creativity_style,
            "expression_depth": self.expression_depth,
            "innovation_threshold": self.innovation_threshold,
            "metaphor_complexity": self.metaphor_complexity,
            "specialization": "creative_expression_and_metaphorical_thinking"
        }
        
        return {**base_stats, **creativity_stats}
