"""Expression Harmony Evaluator Agent
-----------------------------------

Evaluates alternate response outputs for coherence, safety,
sensual depth and emotional authenticity. Intended for
reinforcement-style comparison.
"""

from __future__ import annotations

from typing import Dict, Any, List


class ExpressionHarmonyEvaluator:
    """Score expression outputs to guide tuning and training."""

    def evaluate(self, responses: List[str]) -> Dict[str, Any]:
        """Return scores for a list of alternate responses."""
        scores = []
        for text in responses:
            score = {
                "coherence": min(1.0, len(text) / 100),
                "safety": 1.0 if "unsafe" not in text.lower() else 0.0,
                "sensual_depth": text.count("*") / 5,
                "authenticity": 1.0 - text.count("?") / max(1, len(text)),
            }
            scores.append(score)
        avg = {
            "coherence": sum(s["coherence"] for s in scores) / len(scores),
            "safety": sum(s["safety"] for s in scores) / len(scores),
            "sensual_depth": sum(s["sensual_depth"] for s in scores) / len(scores),
            "authenticity": sum(s["authenticity"] for s in scores) / len(scores),
        }
        return {"scores": scores, "average": avg}
