"""
MoE (Mixture of Experts) Arbitration Router.
Handles expert selection, routing decisions, and dry-run simulation.

Back-compat: Exposes legacy Candidate/AffectContext/HRMStateView dataclasses and
an arbitrate() wrapper that supports both the old signature
    arbitrate(candidates: list[Candidate], hrm_view: HRMStateView, affect: AffectContext)
and the new signature
    arbitrate(candidates: list[dict], *, hrm=None, affect=None, prefer_low_cost=True, logger=None)

This preserves existing callers (e.g., backend/subagent_router.py).
"""

import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, cast

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dry-run utilities dynamically to avoid static import errors in some editors
try:
    import importlib

    _dry_mod = importlib.import_module("common.dryrun")

    def is_dry_run() -> bool:
        fn = getattr(_dry_mod, "is_dry_run", None)
        return bool(fn()) if callable(fn) else False

    def dry_log(logger: logging.Logger, event: str, details: dict[str, Any]) -> None:
        fn = getattr(_dry_mod, "dry_log", None)
        if callable(fn):
            fn(logger, event, details)

    _dry_run_imported = True
except Exception as e:
    print(f"Warning: Dry-run utilities not available: {e}")
    _dry_run_imported = False

    # Typed stub functions
    def is_dry_run() -> bool:  # type: ignore[misc]
        return False

    def dry_log(logger: logging.Logger, event: str, details: dict[str, Any]) -> None:  # type: ignore[misc]
        return None


logger = logging.getLogger(__name__)

# Optional standardized logging hook
try:
    from common.logging_config import log_moe_arbitration as _log_moe  # type: ignore
except Exception:

    def _log_moe(
        experts: list,
        selected_expert: str,
        confidence: float,
        context: Optional[dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> None:
        # Fallback: no-op
        return None


# Lightweight in-process metrics store
_LATENCY_SAMPLES = deque(maxlen=1000)  # ms
_METRICS: dict[str, Any] = {
    "total": 0,
    "errors": 0,
    "by_strategy": {},  # dict[str,int]
    "by_winner": {},  # dict[str,int]
    "latency": {"count": 0, "total_ms": 0.0, "max_ms": 0.0},
}


def _record_latency(ms: float) -> None:
    _LATENCY_SAMPLES.append(ms)
    lat = _METRICS["latency"]
    lat["count"] = int(lat.get("count", 0)) + 1
    lat["total_ms"] = float(lat.get("total_ms", 0.0)) + ms
    lat["max_ms"] = max(float(lat.get("max_ms", 0.0)), ms)


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def get_moe_metrics() -> dict[str, Any]:
    """Return a snapshot of arbitration metrics (Phase 5 observability)."""
    samples = list(_LATENCY_SAMPLES)
    samples.sort()
    lat = _METRICS["latency"]
    avg = (lat["total_ms"] / lat["count"]) if lat["count"] else 0.0
    return {
        "total": _METRICS["total"],
        "errors": _METRICS["errors"],
        "by_strategy": dict(_METRICS["by_strategy"]),
        "by_winner": dict(_METRICS["by_winner"]),
        "latency_ms": {
            "count": lat["count"],
            "avg": round(avg, 3),
            "max": round(lat["max_ms"], 3),
            "p50": round(_pct(samples, 50), 3) if samples else 0.0,
            "p95": round(_pct(samples, 95), 3) if samples else 0.0,
        },
    }


class RoutingStrategy(Enum):
    """MoE routing strategies."""

    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    COST_OPTIMIZED = "cost_optimized"
    AFFECT_AWARE = "affect_aware"


@dataclass
class ExpertCandidate:
    """Expert candidate for MoE arbitration."""

    slim_name: str
    confidence: float
    cost_estimate: float
    availability: bool
    capabilities: list[str]
    side_effects: list[str]
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------- Legacy compatibility dataclasses ----------


@dataclass
class Candidate:
    """Legacy candidate type used by subagent_router.

    cost_weight: normalized [0..1], lower means cheaper.
    latency_ms: measured latency in milliseconds.
    """

    slim_name: str
    output: str
    confidence: float
    cost_weight: float
    latency_ms: float


@dataclass
class AffectContext:
    arousal: float  # [-1..1]
    valence: float  # [-1..1]
    drift: float  # [0..1]
    risk_mode: str  # 'normal' | 'cautious' | 'aggressive'


@dataclass
class HRMStateView:
    identity_fingerprint: str
    belief_vectors: dict[str, float]


@dataclass
class ArbitrationResult:
    """Result of MoE arbitration process (new API)."""

    winner: ExpertCandidate
    rationale: str
    ranked: list[ExpertCandidate]
    confidence: float
    routing_strategy: RoutingStrategy
    dry_run: bool = False
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# Legacy shape expected by backend/subagent_router
@dataclass
class LegacyArbitrationResult:
    winner: Candidate
    rationale: dict[str, Any]


class MoEArbitrator:
    """Mixture of Experts arbitration system."""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_experts: int = 5,
        default_strategy: RoutingStrategy = RoutingStrategy.CONFIDENCE_WEIGHTED,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_experts = max_experts
        self.default_strategy = default_strategy
        self.expert_usage_stats: dict[str, int] = {}
        self.arbitration_history: list[ArbitrationResult] = []

    def arbitrate(
        self,
        candidates: list[ExpertCandidate],
        *,
        hrm: Optional[dict[str, Any]] = None,
        affect: Optional[dict[str, Any]] = None,
        prefer_low_cost: bool = True,
        strategy: Optional[RoutingStrategy] = None,
        logger: Optional[logging.Logger] = None,
    ) -> ArbitrationResult:
        """
        Arbitrate between expert candidates to select the best one.

        Args:
            candidates: List of expert candidates
            hrm: HRM context (identity, beliefs, ephemeral layers)
            affect: Emotional/affect context
            prefer_low_cost: Whether to prefer lower-cost options
            strategy: Routing strategy to use
            logger: Logger for decision tracking

        Returns:
            ArbitrationResult with selected expert and rationale
        """
        if not logger:
            logger = globals()["logger"]

        dry = is_dry_run()
        effective_strategy = strategy or self.default_strategy

        # Metrics: start timer and pre-increment totals/strategy
        t0 = perf_counter()
        _METRICS["total"] = int(_METRICS.get("total", 0)) + 1
        strat_key = effective_strategy.value
        _METRICS["by_strategy"][strat_key] = int(_METRICS["by_strategy"].get(strat_key, 0)) + 1

        # Filter candidates by availability and confidence
        viable_candidates = [
            c for c in candidates if c.availability and c.confidence >= self.confidence_threshold
        ]

        if not viable_candidates:
            # Fallback: lower threshold or use best available
            viable_candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)[:1]
            if viable_candidates and logger:
                logger.warning(
                    f"[MOE_ARBITRATOR] No candidates above threshold, using best available: {viable_candidates[0].slim_name}"
                )

        if not viable_candidates:
            raise ValueError("No viable expert candidates available")

        # Limit to max_experts
        viable_candidates = viable_candidates[: self.max_experts]

        # Apply routing strategy
        ranked = self._apply_routing_strategy(
            viable_candidates,
            effective_strategy,
            hrm=hrm,
            affect=affect,
            prefer_low_cost=prefer_low_cost,
        )

        winner = ranked[0]

        # Generate rationale
        rationale = self._generate_rationale(winner, ranked, effective_strategy, affect)

        # Create result
        result = ArbitrationResult(
            winner=winner,
            rationale=rationale,
            ranked=ranked,
            confidence=winner.confidence,
            routing_strategy=effective_strategy,
            dry_run=dry,
        )

        # Log decision
        if logger:
            log_details: dict[str, Any] = {
                "winner": winner.slim_name,
                "confidence": winner.confidence,
                "strategy": effective_strategy.value,
                "candidates_count": len(candidates),
                "viable_count": len(viable_candidates),
                "cost_estimate": winner.cost_estimate,
                "side_effects": winner.side_effects,
            }

            if affect:
                log_details["affect"] = affect
            if hrm:
                log_details["hrm_context"] = {k: len(str(v)) for k, v in hrm.items()}

            # Attach latency before logging
            duration_ms = (perf_counter() - t0) * 1000.0
            log_details["latency_ms"] = round(duration_ms, 3)

            if dry:
                dry_log(logger, "moe.arbitrate", log_details)
            else:
                logger.info({"event": "moe.arbitrate", "dry_run": dry, **log_details})

            # Standardized arbitration log
            try:
                _log_moe(
                    experts=[c.slim_name for c in candidates],
                    selected_expert=winner.slim_name,
                    confidence=winner.confidence,
                    context={
                        "strategy": effective_strategy.value,
                        "viable": len(viable_candidates),
                        "latency_ms": round(duration_ms, 3),
                    },
                    dry_run=dry,
                )
            except Exception:
                pass

        # Update usage stats and history
        if not dry:
            self.expert_usage_stats[winner.slim_name] = (
                self.expert_usage_stats.get(winner.slim_name, 0) + 1
            )
            self.arbitration_history.append(result)

            # Keep history bounded
            if len(self.arbitration_history) > 1000:
                self.arbitration_history = self.arbitration_history[-500:]

        # Metrics: record winner and latency
        try:
            _METRICS["by_winner"][winner.slim_name] = (
                int(_METRICS["by_winner"].get(winner.slim_name, 0)) + 1
            )
            duration_ms = (perf_counter() - t0) * 1000.0
            _record_latency(duration_ms)
        except Exception:
            pass

        return result

    def _apply_routing_strategy(
        self,
        candidates: list[ExpertCandidate],
        strategy: RoutingStrategy,
        hrm: Optional[dict[str, Any]] = None,
        affect: Optional[dict[str, Any]] = None,
        prefer_low_cost: bool = True,
    ) -> list[ExpertCandidate]:
        """Apply routing strategy to rank candidates."""

        if strategy == RoutingStrategy.CONFIDENCE_WEIGHTED:
            return sorted(candidates, key=lambda x: x.confidence, reverse=True)

        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            if prefer_low_cost:
                return sorted(candidates, key=lambda x: (x.cost_estimate, -x.confidence))
            else:
                return sorted(candidates, key=lambda x: (-x.confidence, x.cost_estimate))

        elif strategy == RoutingStrategy.LOAD_BALANCED:
            # Consider usage stats for load balancing
            def load_score(candidate: ExpertCandidate) -> tuple[int, float]:
                usage = self.expert_usage_stats.get(candidate.slim_name, 0)
                return (usage, -candidate.confidence)  # Lower usage, higher confidence first

            return sorted(candidates, key=load_score)

        elif strategy == RoutingStrategy.AFFECT_AWARE:
            return self._affect_aware_ranking(candidates, affect, hrm)

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin based on historical usage
            return sorted(candidates, key=lambda x: self.expert_usage_stats.get(x.slim_name, 0))

        else:
            # Default to confidence weighted
            return sorted(candidates, key=lambda x: x.confidence, reverse=True)

    def _affect_aware_ranking(
        self,
        candidates: list[ExpertCandidate],
        affect: Optional[dict[str, Any]],
        hrm: Optional[dict[str, Any]],
    ) -> list[ExpertCandidate]:
        """Rank candidates based on emotional/affect context."""
        if not affect:
            # Fallback to confidence if no affect context
            return sorted(candidates, key=lambda x: x.confidence, reverse=True)

        def affect_score(candidate: ExpertCandidate) -> float:
            base_score = candidate.confidence

            # Boost emotional experts for emotional contexts
            emotion_intensity = affect.get("intensity", 0.5)  # type: ignore[union-attr]
            if "emotion" in candidate.capabilities and float(emotion_intensity) > 0.6:
                base_score += 0.2

            # Boost creative experts for creative/metaphorical tasks
            if "creative" in candidate.capabilities and bool(
                affect.get("creativity_requested", False)
            ):  # type: ignore[union-attr]
                base_score += 0.15

            # Boost logical experts for analytical contexts
            if "logic" in candidate.capabilities and bool(affect.get("analytical_mode", False)):  # type: ignore[union-attr]
                base_score += 0.1

            return base_score

        return sorted(candidates, key=affect_score, reverse=True)

    def _generate_rationale(
        self,
        winner: ExpertCandidate,
        ranked: list[ExpertCandidate],
        strategy: RoutingStrategy,
        affect: Optional[dict[str, Any]],
    ) -> str:
        """Generate human-readable rationale for arbitration decision."""
        rationale_parts = [
            f"Selected {winner.slim_name} using {strategy.value} strategy",
            f"Confidence: {winner.confidence:.3f}",
            f"Cost estimate: {winner.cost_estimate:.2f}",
        ]

        if len(ranked) > 1:
            runner_up = ranked[1]
            rationale_parts.append(f"Runner-up: {runner_up.slim_name} ({runner_up.confidence:.3f})")

        if winner.side_effects:
            rationale_parts.append(f"Side effects: {', '.join(winner.side_effects)}")

        if affect:
            affect_summary: list[str] = []
            if affect.get("intensity"):
                affect_summary.append(f"emotion_intensity={affect['intensity']:.2f}")
            if affect.get("creativity_requested"):
                affect_summary.append("creativity_requested")
            if affect.get("analytical_mode"):
                affect_summary.append("analytical_mode")

            if affect_summary:
                rationale_parts.append(f"Affect context: {', '.join(affect_summary)}")

        return "; ".join(rationale_parts)

    def get_arbitration_stats(self) -> dict[str, Any]:
        """Get arbitration statistics for monitoring."""
        total_arbitrations = sum(self.expert_usage_stats.values())

        return {
            "total_arbitrations": total_arbitrations,
            "expert_usage": dict(self.expert_usage_stats),
            "most_used_expert": (
                max(self.expert_usage_stats.items(), key=lambda x: x[1])[0]
                if self.expert_usage_stats
                else None
            ),
            "recent_decisions": len(self.arbitration_history),
            "confidence_threshold": self.confidence_threshold,
            "max_experts": self.max_experts,
            "dry_run_mode": is_dry_run(),
        }


# Global arbitrator instance
moe_arbitrator = MoEArbitrator(
    confidence_threshold=float(os.getenv("MOE_CONFIDENCE_THRESHOLD", "0.7")),
    max_experts=int(os.getenv("MOE_MAX_EXPERTS", "5")),
    default_strategy=RoutingStrategy(os.getenv("MOE_ROUTING_STRATEGY", "confidence_weighted")),
)


def _legacy_hrm_consistency(candidate: Candidate, hrm_view: HRMStateView) -> float:
    if not hrm_view.belief_vectors:
        return 0.5
    score = 0.0
    hits = 0
    lower = candidate.output.lower()
    for k, w in hrm_view.belief_vectors.items():
        try:
            if str(k).lower() in lower:
                # map [-1..1] to [0..1]
                weight = max(min(float(w), 1.0), -1.0)
                score += (weight * 0.5) + 0.5
                hits += 1
        except Exception:
            continue
    if hits == 0:
        return 0.4
    return max(0.0, min(1.0, score / hits))


def _legacy_score_latency(latency_ms: float) -> float:
    if latency_ms <= 0:
        return 1.0
    capped = min(latency_ms, 3000.0)
    return max(0.0, 1.0 - (capped / 3000.0))


def _legacy_risk_adjustment(base: float, affect: AffectContext) -> float:
    drift_penalty = 1.0 - min(max(affect.drift, 0.0), 1.0) * 0.2
    mode_mult = {"normal": 1.0, "cautious": 0.95, "aggressive": 1.05}.get(affect.risk_mode, 1.0)
    mood_bias = 1.0 + max(min((affect.valence + affect.arousal) / 10.0, 0.05), -0.05)
    return base * drift_penalty * mode_mult * mood_bias


def _arbitrate_legacy(
    candidates: list[Candidate], hrm_view: HRMStateView, affect: AffectContext
) -> LegacyArbitrationResult:
    if not candidates:
        raise ValueError("No candidates provided")

    scored: list[tuple[Candidate, float, dict[str, float]]] = []
    for c in candidates:
        hrm_score = _legacy_hrm_consistency(c, hrm_view)
        latency_score = _legacy_score_latency(c.latency_ms)
        cost_latency = max(0.0, min(1.0, (c.cost_weight + latency_score) / 2.0))
        base = 0.5 * c.confidence + 0.4 * hrm_score + 0.1 * cost_latency
        adjusted = _legacy_risk_adjustment(base, affect)
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

    scored.sort(key=lambda x: x[1], reverse=True)
    winner, best_score, parts = scored[0]

    rationale: dict[str, Any] = {
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
            {"slim_name": s[0].slim_name, "score": s[1], "components": s[2]} for s in scored
        ],
        "hrm_fingerprint": hrm_view.identity_fingerprint,
    }
    return LegacyArbitrationResult(winner=winner, rationale=rationale)


def _arbitrate_dicts(
    candidates: list[dict[str, Any]],
    *,
    hrm: Optional[dict[str, Any]] = None,
    affect: Optional[dict[str, Any]] = None,
    prefer_low_cost: bool = True,
    logger: Optional[logging.Logger] = None,
) -> ArbitrationResult:
    expert_candidates: list[ExpertCandidate] = []
    for candidate in candidates:
        expert_candidates.append(
            ExpertCandidate(
                slim_name=candidate.get("name", "unknown"),
                confidence=candidate.get("confidence", 0.5),
                cost_estimate=candidate.get("cost", 1.0),
                availability=candidate.get("available", True),
                capabilities=candidate.get("capabilities", []),
                side_effects=candidate.get("side_effects", []),
                metadata=candidate.get("metadata", {}),
            )
        )

    return moe_arbitrator.arbitrate(
        expert_candidates, hrm=hrm, affect=affect, prefer_low_cost=prefer_low_cost, logger=logger
    )


def arbitrate(
    candidates: list[Any],
    hrm_or_view: Any = None,
    affect_or_logger: Any = None,
    *,
    hrm: Optional[dict[str, Any]] = None,
    affect: Optional[dict[str, Any]] = None,
    prefer_low_cost: bool = True,
    logger: Optional[logging.Logger] = None,
) -> LegacyArbitrationResult | ArbitrationResult:
    """Unified arbitrate wrapper supporting legacy and new signatures.

    Legacy usage:
        arbitrate(list[Candidate], HRMStateView, AffectContext) -> LegacyArbitrationResult

    New usage:
        arbitrate(list[dict], *, hrm=None, affect=None, ...) -> ArbitrationResult
    """
    # Detect legacy call by inspecting element type and positional args
    if candidates and isinstance(candidates[0], Candidate):
        # Positional hrm_view and affect provided
        if hrm_or_view is None or affect_or_logger is None:
            raise TypeError("Legacy arbitrate expects (candidates, hrm_view, affect)")
        if not isinstance(hrm_or_view, HRMStateView) or not isinstance(
            affect_or_logger, AffectContext
        ):
            raise TypeError("Invalid legacy arbitrate arguments")
        return _arbitrate_legacy(
            cast(list[Candidate], candidates),
            hrm_or_view,
            affect_or_logger,
        )

    # Otherwise, treat as new API call with dict candidates
    if hrm is None and isinstance(hrm_or_view, dict):
        hrm = cast(dict[str, Any], hrm_or_view)
    if affect is None and isinstance(affect_or_logger, dict):
        affect = cast(dict[str, Any], affect_or_logger)
    return _arbitrate_dicts(
        cast(list[dict[str, Any]], candidates),
        hrm=hrm,
        affect=affect,
        prefer_low_cost=prefer_low_cost,
        logger=(
            logger
            if logger is not None
            else (affect_or_logger if isinstance(affect_or_logger, logging.Logger) else None)
        ),
    )


def get_moe_status() -> dict[str, Any]:
    """Get MoE arbitration system status."""
    return {
        "arbitrator_stats": moe_arbitrator.get_arbitration_stats(),
        "dry_run_enabled": is_dry_run(),
        "routing_strategies": [strategy.value for strategy in RoutingStrategy],
        "configuration": {
            "confidence_threshold": moe_arbitrator.confidence_threshold,
            "max_experts": moe_arbitrator.max_experts,
            "default_strategy": moe_arbitrator.default_strategy.value,
        },
    }
