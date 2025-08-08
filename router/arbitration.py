"""
MoE (Mixture of Experts) Arbitration Router.
Handles expert selection, routing decisions, and dry-run simulation.
"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dry-run utilities
try:
    from common.dryrun import dry_guard, dry_log, format_dry_run_response, is_dry_run

    DRY_RUN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dry-run utilities not available: {e}")
    DRY_RUN_AVAILABLE = False

    # Create stub functions
    def is_dry_run():
        return False

    def dry_log(logger, event, details):
        pass

    def dry_guard(logger, event, details):
        from contextlib import contextmanager

        @contextmanager
        def stub():
            yield False

        return stub()


logger = logging.getLogger(__name__)


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
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ArbitrationResult:
    """Result of MoE arbitration process."""

    winner: ExpertCandidate
    rationale: str
    ranked: list[ExpertCandidate]
    confidence: float
    routing_strategy: RoutingStrategy
    dry_run: bool = False
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


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

        # Filter candidates by availability and confidence
        viable_candidates = [
            c for c in candidates if c.availability and c.confidence >= self.confidence_threshold
        ]

        if not viable_candidates:
            # Fallback: lower threshold or use best available
            viable_candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)[:1]
            if viable_candidates:
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
            log_details = {
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

            (
                dry_log(logger, "moe.arbitrate", log_details)
                if dry
                else logger.info({"event": "moe.arbitrate", "dry_run": dry, **log_details})
            )

        # Update usage stats and history
        if not dry:
            self.expert_usage_stats[winner.slim_name] = (
                self.expert_usage_stats.get(winner.slim_name, 0) + 1
            )
            self.arbitration_history.append(result)

            # Keep history bounded
            if len(self.arbitration_history) > 1000:
                self.arbitration_history = self.arbitration_history[-500:]

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
            def load_score(candidate):
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

        def affect_score(candidate):
            base_score = candidate.confidence

            # Boost emotional experts for emotional contexts
            emotion_intensity = affect.get("intensity", 0.5)
            if "emotion" in candidate.capabilities and emotion_intensity > 0.6:
                base_score += 0.2

            # Boost creative experts for creative/metaphorical tasks
            if "creative" in candidate.capabilities and affect.get("creativity_requested", False):
                base_score += 0.15

            # Boost logical experts for analytical contexts
            if "logic" in candidate.capabilities and affect.get("analytical_mode", False):
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
            affect_summary = []
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


def arbitrate(
    candidates: list[dict[str, Any]],
    *,
    hrm: Optional[dict[str, Any]] = None,
    affect: Optional[dict[str, Any]] = None,
    prefer_low_cost: bool = True,
    logger: Optional[logging.Logger] = None,
) -> ArbitrationResult:
    """
    Main arbitration function for MoE routing.

    Args:
        candidates: List of candidate expert dictionaries
        hrm: HRM context
        affect: Emotional/affect context
        prefer_low_cost: Whether to prefer lower-cost options
        logger: Logger for decision tracking

    Returns:
        ArbitrationResult with selected expert and rationale
    """
    # Convert dict candidates to ExpertCandidate objects
    expert_candidates = []
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
