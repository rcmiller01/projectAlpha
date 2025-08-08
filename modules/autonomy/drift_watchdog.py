"""
Drift Watchdog: tracks rolling average of drift deltas and flags sustained breaches.

update(drift)->status:
- Maintains a rolling window (seconds) of drift values.
- Returns {breach: bool, avg: float, window_n: int}
- Emits metric affect_spikes_blocked_total and logs affect.watchdog.breach on sustained breach.

Config via env:
- DRIFT_WINDOW_SECONDS (int, default 10)
- DRIFT_BREACH_THRESHOLD (float, default 0.7)
- DRIFT_BREACH_HOLDDOWN (int, default 5)  # seconds to avoid flapping
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, Deque, Dict, Tuple

logger = logging.getLogger(__name__)


class DriftWatchdog:
    def __init__(self) -> None:
        self.window_seconds = int(os.getenv("DRIFT_WINDOW_SECONDS", 10))
        self.threshold = float(os.getenv("DRIFT_BREACH_THRESHOLD", 0.7))
        self.holddown = int(os.getenv("DRIFT_BREACH_HOLDDOWN", 5))
        self.buffer: Deque[Tuple[float, float]] = deque()  # (ts, drift)
        self._last_breach_ts: float = 0.0
        self.metrics: Dict[str, int] = {"affect_watchdog_breaches_total": 0}

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def update(self, drift: float) -> Dict[str, Any]:
        now = time.time()
        self._prune(now)
        self.buffer.append((now, float(drift)))
        if not self.buffer:
            return {"breach": False, "avg": 0.0, "window_n": 0}

        avg = sum(v for _, v in self.buffer) / max(1, len(self.buffer))
        breach = avg >= self.threshold

        # Holddown logic: log once per holddown window
        if breach and (now - self._last_breach_ts) >= self.holddown:
            self._last_breach_ts = now
            self.metrics["affect_watchdog_breaches_total"] += 1
            try:
                logger.warning(
                    json.dumps(
                        {
                            "event": "affect.watchdog.breach",
                            "avg": avg,
                            "window_n": len(self.buffer),
                            "threshold": self.threshold,
                        }
                    )
                )
            except Exception:
                pass

        return {"breach": breach, "avg": avg, "window_n": len(self.buffer)}
