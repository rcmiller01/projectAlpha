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
- HRM identity and beliefs validation

Author: ProjectAlpha Team
"""

import logging
from typing import Any, Dict, List, Optional

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
            agent_id=agent_id or "deduction_agent",
        )

        # DeductionAgent-specific configuration
        self.reasoning_style = "deductive"
        self.confidence_threshold = 0.8  # High confidence for logical reasoning
        self.max_proof_steps = 10

        # HRM validation settings
        self.hrm_validation_enabled = True
        self.identity_validation_threshold = 0.7
        self.beliefs_validation_threshold = 0.6

        logger.info(
            f"DeductionAgent {self.agent_id} initialized for logical reasoning with HRM validation"
        )

    def prove(self, statement: str, premises: Optional[list[str]] = None) -> str:
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

        logger.debug("DeductionAgent analyzing argument validity")
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

        logger.debug("DeductionAgent solving mathematical problem")
        return self.run(full_prompt, depth=1, use_tools=True)

    def deduce_from_facts(self, facts: list[str], question: Optional[str] = None) -> str:
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

        logger.debug("DeductionAgent performing deductive reasoning")
        return self.run(full_prompt, depth=2, use_tools=True)

    def _evaluate_tool_usage(self, prompt: str, context: list[dict[str, Any]]) -> list[str]:
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

    def _update_memory(
        self, prompt: str, response: str, context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
                        "solution_length": len(response),
                    },
                }

                self.memory.add_fact(
                    fact["subject"], fact["predicate"], fact["object"], fact["metadata"]
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
                        "proof_complexity": "high" if len(response) > 500 else "medium",
                    },
                }

                self.memory.add_fact(
                    fact["subject"], fact["predicate"], fact["object"], fact["metadata"]
                )
                memory_updates.append(fact)

        except Exception as e:
            logger.warning(f"Enhanced memory update failed: {e}")

        return memory_updates

    def validate_against_hrm_identity(
        self, deduction: str, context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Validate deduction against HRM identity layer.

        Args:
            deduction: The logical deduction to validate
            context: Optional context for validation

        Returns:
            Validation result with identity compatibility score
        """
        if not self.hrm_validation_enabled:
            return {"validation_enabled": False, "identity_score": 1.0, "status": "skipped"}

        try:
            # Extract identity-related facts from memory
            identity_facts = []
            if hasattr(self.memory, "query_facts"):
                identity_facts = self.memory.query_facts(
                    subject_pattern="identity", predicate_pattern="defines|represents|characterizes"
                )

            validation_result = {
                "validation_enabled": True,
                "deduction": deduction[:100] + "..." if len(deduction) > 100 else deduction,
                "identity_facts_count": len(identity_facts),
                "identity_score": 0.0,
                "compatibility": "unknown",
                "conflicts": [],
                "timestamp": str(context.get("timestamp")) if context else None,
            }

            # Analyze identity compatibility
            deduction_lower = deduction.lower()
            identity_keywords = ["self", "identity", "purpose", "mission", "values", "core"]

            identity_relevance = sum(
                1 for keyword in identity_keywords if keyword in deduction_lower
            )
            base_score = min(1.0, identity_relevance * 0.2)

            # Check for identity conflicts
            conflict_indicators = ["contradict", "oppose", "violate", "conflict", "deny"]
            conflicts_found = [
                indicator for indicator in conflict_indicators if indicator in deduction_lower
            ]

            if conflicts_found:
                validation_result["conflicts"] = conflicts_found
                validation_result["identity_score"] = max(
                    0.0, base_score - len(conflicts_found) * 0.3
                )
            else:
                validation_result["identity_score"] = min(1.0, base_score + 0.5)

            # Determine compatibility status
            if validation_result["identity_score"] >= self.identity_validation_threshold:
                validation_result["compatibility"] = "compatible"
            elif validation_result["identity_score"] >= 0.3:
                validation_result["compatibility"] = "partially_compatible"
            else:
                validation_result["compatibility"] = "incompatible"

            logger.debug(
                f"HRM identity validation - Score: {validation_result['identity_score']}, Status: {validation_result['compatibility']}"
            )
            return validation_result

        except Exception as e:
            logger.error(f"HRM identity validation failed: {e}")
            return {
                "validation_enabled": True,
                "identity_score": 0.0,
                "status": "error",
                "error": str(e),
            }

    def validate_against_hrm_beliefs(
        self, deduction: str, context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Validate deduction against HRM beliefs layer.

        Args:
            deduction: The logical deduction to validate
            context: Optional context for validation

        Returns:
            Validation result with beliefs compatibility score
        """
        if not self.hrm_validation_enabled:
            return {"validation_enabled": False, "beliefs_score": 1.0, "status": "skipped"}

        try:
            # Extract belief-related facts from memory
            belief_facts = []
            if hasattr(self.memory, "query_facts"):
                belief_facts = self.memory.query_facts(
                    subject_pattern="belief|conviction|principle",
                    predicate_pattern="holds|maintains|believes",
                )

            validation_result = {
                "validation_enabled": True,
                "deduction": deduction[:100] + "..." if len(deduction) > 100 else deduction,
                "belief_facts_count": len(belief_facts),
                "beliefs_score": 0.0,
                "alignment": "unknown",
                "contradictions": [],
                "timestamp": str(context.get("timestamp")) if context else None,
            }

            # Analyze beliefs alignment
            deduction_lower = deduction.lower()
            belief_keywords = ["believe", "truth", "principle", "conviction", "ethics", "moral"]

            belief_relevance = sum(1 for keyword in belief_keywords if keyword in deduction_lower)
            base_score = min(1.0, belief_relevance * 0.15)

            # Check for belief contradictions
            contradiction_indicators = ["false", "wrong", "invalid", "reject", "disagree"]
            contradictions_found = [
                indicator for indicator in contradiction_indicators if indicator in deduction_lower
            ]

            if contradictions_found:
                validation_result["contradictions"] = contradictions_found
                validation_result["beliefs_score"] = max(
                    0.0, base_score - len(contradictions_found) * 0.25
                )
            else:
                validation_result["beliefs_score"] = min(1.0, base_score + 0.6)

            # Determine alignment status
            if validation_result["beliefs_score"] >= self.beliefs_validation_threshold:
                validation_result["alignment"] = "aligned"
            elif validation_result["beliefs_score"] >= 0.3:
                validation_result["alignment"] = "partially_aligned"
            else:
                validation_result["alignment"] = "misaligned"

            logger.debug(
                f"HRM beliefs validation - Score: {validation_result['beliefs_score']}, Status: {validation_result['alignment']}"
            )
            return validation_result

        except Exception as e:
            logger.error(f"HRM beliefs validation failed: {e}")
            return {
                "validation_enabled": True,
                "beliefs_score": 0.0,
                "status": "error",
                "error": str(e),
            }

    def validate_deduction_against_hrm(
        self, deduction: str, context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Comprehensive HRM validation combining identity and beliefs layers.

        Args:
            deduction: The logical deduction to validate
            context: Optional context for validation

        Returns:
            Combined validation result
        """
        identity_validation = self.validate_against_hrm_identity(deduction, context)
        beliefs_validation = self.validate_against_hrm_beliefs(deduction, context)

        combined_result = {
            "deduction": deduction[:100] + "..." if len(deduction) > 100 else deduction,
            "identity_validation": identity_validation,
            "beliefs_validation": beliefs_validation,
            "overall_score": 0.0,
            "overall_status": "unknown",
            "recommendations": [],
            "validation_timestamp": str(context.get("timestamp")) if context else None,
        }

        # Calculate overall score (weighted average)
        identity_weight = 0.6
        beliefs_weight = 0.4

        identity_score = identity_validation.get("identity_score", 0.0)
        beliefs_score = beliefs_validation.get("beliefs_score", 0.0)

        combined_result["overall_score"] = (identity_score * identity_weight) + (
            beliefs_score * beliefs_weight
        )

        # Determine overall status
        if combined_result["overall_score"] >= 0.7:
            combined_result["overall_status"] = "validated"
        elif combined_result["overall_score"] >= 0.5:
            combined_result["overall_status"] = "partially_validated"
            combined_result["recommendations"].append(
                "Consider reviewing deduction for better alignment with HRM layers"
            )
        else:
            combined_result["overall_status"] = "validation_failed"
            combined_result["recommendations"].append(
                "Deduction requires significant revision for HRM compatibility"
            )

        # Add specific recommendations based on validation results
        if identity_validation.get("compatibility") == "incompatible":
            combined_result["recommendations"].append(
                "Review deduction for identity layer compatibility"
            )

        if beliefs_validation.get("alignment") == "misaligned":
            combined_result["recommendations"].append(
                "Align deduction with core beliefs and principles"
            )

        logger.info(
            f"HRM validation complete - Overall score: {combined_result['overall_score']}, Status: {combined_result['overall_status']}"
        )
        return combined_result

    def get_reasoning_stats(self) -> dict[str, Any]:
        """Get statistics specific to logical reasoning performance."""
        base_stats = self.get_status()

        reasoning_stats = {
            "reasoning_style": self.reasoning_style,
            "confidence_threshold": self.confidence_threshold,
            "max_proof_steps": self.max_proof_steps,
            "specialization": "logical_reasoning_and_mathematical_deduction",
            "hrm_validation_enabled": self.hrm_validation_enabled,
            "identity_validation_threshold": self.identity_validation_threshold,
            "beliefs_validation_threshold": self.beliefs_validation_threshold,
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
