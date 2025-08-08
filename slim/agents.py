"""
Registered SLiM agent implementations (stubs) decorated with contracts.
These provide simple, fast behaviors and enable health/metrics reporting.
"""
from __future__ import annotations

from typing import Any, Optional
import time

from .sdk import (
    LOGIC_HIGH_CONTRACT,
    EMOTION_VALENCE_CONTRACT,
    CREATIVE_METAPHOR_CONTRACT,
    contract_validator,
)


@contract_validator(LOGIC_HIGH_CONTRACT)
def logic_high(
    reasoning_task: str,
    context: dict[str, Any],
    complexity: str = "moderate",
    simulate_ms: Optional[int] = 0,
) -> dict[str, Any]:
    if simulate_ms and simulate_ms > 0:
        time.sleep(simulate_ms / 1000.0)
    steps = [f"Analyze task: {reasoning_task}", f"Consider context keys: {list(context.keys())}"]
    assumptions = ["Inputs are sanitized", f"Complexity is {complexity}"]
    reasoning = f"Task '{reasoning_task}' addressed with {complexity} depth."
    return {
        "reasoning": reasoning,
        "confidence": 0.82,
        "steps": steps,
        "assumptions": assumptions,
    }


@contract_validator(EMOTION_VALENCE_CONTRACT)
def emotion_valence(
    text: str,
    context: dict[str, Any],
    previous_state: Optional[dict[str, Any]] = None,
    simulate_ms: Optional[int] = 0,
) -> dict[str, Any]:
    if simulate_ms and simulate_ms > 0:
        time.sleep(simulate_ms / 1000.0)
    # naive rule: presence of "good" vs "bad"
    t = text.lower()
    valence = 0.5 if "good" in t else (-0.3 if "bad" in t else 0.0)
    emotion = "positive" if valence > 0 else ("negative" if valence < 0 else "neutral")
    intensity = min(1.0, abs(valence) + 0.2)
    tags = [emotion]
    return {
        "emotion": emotion,
        "valence": valence,
        "intensity": intensity,
        "emotional_tags": tags,
    }


@contract_validator(CREATIVE_METAPHOR_CONTRACT)
def creative_metaphor(
    concept: str,
    target_audience: str,
    style: str | None = None,
    complexity: str | None = None,
    simulate_ms: Optional[int] = 0,
) -> dict[str, Any]:
    if simulate_ms and simulate_ms > 0:
        time.sleep(simulate_ms / 1000.0)
    metaphor = f"{concept} is a lighthouse for the {target_audience}"
    explanation = (
        f"Framing '{concept}' as a lighthouse conveys guidance and clarity tailored to {target_audience}."
    )
    effectiveness = 0.76
    alternatives = [f"{concept} as a compass", f"{concept} as a bridge"]
    return {
        "metaphor": metaphor,
        "explanation": explanation,
        "effectiveness": effectiveness,
        "alternative_metaphors": alternatives,
    }
