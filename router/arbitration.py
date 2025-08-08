from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class Candidate:
    slim_name: str
    output: str
    confidence: float
    cost_weight: float  # normalized cost score [0..1], lower cost -> higher score
    latency_ms: float   # measured latency in milliseconds


@dataclass
class AffectContext:
    arousal: float      # [-1..1]
    valence: float      # [-1..1]
    drift: float        # [0..1] drift magnitude
    risk_mode: str      # 'normal' | 'cautious' | 'aggressive'


@dataclass
class HRMStateView:
    identity_fingerprint: str
    belief_vectors: Dict[str, float]  # simplified representation


@dataclass
class ArbitrationResult:
    winner: Candidate
    rationale: Dict[str, Any]


def _score_latency(latency_ms: float) -> float:
    # Map latency to [0..1] where faster is better. Cap at 3s.
    if latency_ms <= 0:
        return 1.0
    capped = min(latency_ms, 3000.0)
    return max(0.0, 1.0 - (capped / 3000.0))


def _risk_adjustment(base: float, affect: AffectContext) -> float:
    # Adjust score by risk posture and drift
    drift_penalty = 1.0 - min(max(affect.drift, 0.0), 1.0) * 0.2  # up to -20%
    mode_mult = {
        "normal": 1.0,
        "cautious": 0.95,
        "aggressive": 1.05,
    }.get(affect.risk_mode, 1.0)

    # Valence/arousal can bias slightly (+/- up to 5%)
    mood_bias = 1.0 + max(min((affect.valence + affect.arousal) / 10.0, 0.05), -0.05)

    return base * drift_penalty * mode_mult * mood_bias


def _hrm_consistency(candidate: Candidate, hrm: HRMStateView) -> float:
    # Placeholder: basic cosine-like similarity using belief_vectors keys in output
    if not hrm.belief_vectors:
        return 0.5
    score = 0.0
    hits = 0
    lower = candidate.output.lower()
    for k, w in hrm.belief_vectors.items():
        if k.lower() in lower:
            score += max(min(w, 1.0), -1.0) * 0.5 + 0.5  # map [-1..1] -> [0..1]
            hits += 1
    if hits == 0:
        return 0.4  # slight penalty for no alignment
    return max(0.0, min(1.0, score / hits))


def arbitrate(candidates: List[Candidate], hrm_view: HRMStateView, affect: AffectContext) -> ArbitrationResult:
    """
    Score each candidate with:
      score = 0.5*confidence + 0.4*HRM_consistency + 0.1*cost_latency
    cost_latency combines cost_weight and inverse latency equally.
    Apply risk adjustment, pick max.
    """
    if not candidates:
        raise ValueError("No candidates provided")

    scored: List[Tuple[Candidate, float, Dict[str, float]]] = []
    for c in candidates:
        hrm_score = _hrm_consistency(c, hrm_view)
        latency_score = _score_latency(c.latency_ms)
        cost_latency = max(0.0, min(1.0, (c.cost_weight + latency_score) / 2.0))
        base = 0.5 * c.confidence + 0.4 * hrm_score + 0.1 * cost_latency
        adjusted = _risk_adjustment(base, affect)
        parts = {
            "confidence": c.confidence,
            "hrm_consistency": hrm_score,
            "cost_latency": cost_latency,
            "base": base,
            "adjusted": adjusted,
            "latency_ms": c.latency_ms,
            "cost_weight": c.cost_weight,
        }
        scored.append((c, adjusted, parts))

    # Choose winner
    scored.sort(key=lambda x: x[1], reverse=True)
    winner, best_score, parts = scored[0]

    rationale = {
        "winner": winner.slim_name,
        "best_score": best_score,
        "components": parts,
        "affect": {
            "arousal": affect.arousal,
            "valence": affect.valence,
            "drift": affect.drift,
            "risk_mode": affect.risk_mode,
        },
        "all_scores": [
            {
                "slim_name": s[0].slim_name,
                "score": s[1],
                "components": s[2],
            }
            for s in scored
        ],
        "hrm_fingerprint": hrm_view.identity_fingerprint,
    }
    return ArbitrationResult(winner=winner, rationale=rationale)
