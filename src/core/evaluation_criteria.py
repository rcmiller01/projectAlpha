"""Evaluation criteria enumeration for JudgeAgent."""

from enum import Enum


class EvaluationCriterion(Enum):
    """Criteria used for evaluating generated content."""

    EMOTIONAL_FIDELITY = "emotional_fidelity"
    PERSONA_ALIGNMENT = "persona_alignment"
    NARRATIVE_COHERENCE = "narrative_coherence"
    MEMORY_CONGRUENCE = "memory_congruence"
    ARTISTIC_EXPRESSION = "artistic_expression"
