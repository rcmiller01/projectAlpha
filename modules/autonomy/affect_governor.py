"""
Affect Governor: dampens sudden affect spikes via configurable clamping curves
and tracks per-source cooldowns to prevent repeated surges.

Config via env:
- DRIFT_SCALING_FACTOR (float, default 0.5)
- MAX_PENALTY_THRESHOLD (float, default 0.9)
- AFFECT_COOLDOWN_SECONDS (int, default 2)

apply(state) -> state':
- Expects state to be a dict with keys: {source: str, delta: float}
- Returns dict with keys: {delta, delta_clamped, clamped: bool, penalty: float}
- Increments spikes_blocked metric on clamp.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict

logger = logging.getLogger(__name__)

# Metrics
METRICS: dict[str, int] = defaultdict(int)


class AffectGovernor:
    def __init__(
        self,
        *,
        scaling: float | None = None,
        max_penalty: float | None = None,
        cooldown_seconds: int | None = None,
    ) -> None:
        self.scaling = float(
            os.getenv("DRIFT_SCALING_FACTOR", scaling if scaling is not None else 0.5)
        )
        self.max_penalty = float(
            os.getenv("MAX_PENALTY_THRESHOLD", max_penalty if max_penalty is not None else 0.9)
        )
        self.cooldown_seconds = int(
            os.getenv(
                "AFFECT_COOLDOWN_SECONDS", cooldown_seconds if cooldown_seconds is not None else 2
            )
        )
        self._events: dict[str, deque[float]] = defaultdict(deque)  # per-source timestamps

    def _prune(self, source: str, now: float) -> None:
        dq = self._events[source]
        cutoff = now - self.cooldown_seconds
        while dq and dq[0] < cutoff:
            dq.popleft()

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        source = str(state.get("source", "unknown"))
        delta = float(state.get("delta", 0.0))
        now = time.time()

        # cooldown memory
        self._prune(source, now)
        self._events[source].append(now)
        recent = len(self._events[source])

        # base clamp by scaling curve
        clamped_delta = delta * self.scaling

        # additional penalty if too many events in cooldown window
        penalty = min(self.max_penalty, max(0.0, (recent - 1) / max(1, self.cooldown_seconds * 2)))
        clamped_delta = max(0.0, clamped_delta * (1.0 - penalty))

        clamped = clamped_delta < delta
        if clamped:
            METRICS["affect_spikes_blocked_total"] += 1
            try:
                logger.info(
                    json.dumps(
                        {
                            "event": "affect.governor.clamp",
                            "source": source,
                            "delta": delta,
                            "delta_clamped": clamped_delta,
                            "penalty": penalty,
                            "recent": recent,
                        }
                    )
                )
            except Exception:
                pass

        return {
            "delta": delta,
            "delta_clamped": clamped_delta,
            "clamped": clamped,
            "penalty": penalty,
        }


def get_metrics() -> dict[str, int]:
    return dict(METRICS)
