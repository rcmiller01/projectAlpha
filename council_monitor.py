#!/usr/bin/env python3
"""
Passive runtime monitor for council and core evaluation.

This utility periodically checks quantization tracking logs and prints a concise
status summary. It can be run alongside other services to watch the health of
all active models without consuming significant resources.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from quant_tracking import QuantTracker


class CouncilMonitor:
    """Lightweight monitoring of council/core metrics"""

    def __init__(self, tracking_file: str = "data/quantization_tracking.jsonl", interval: int = 300) -> None:
        self.tracker = QuantTracker(tracking_file)
        self.interval = interval
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("CouncilMonitor")
        self.logger.info("📊 Council Monitor started")

    def summarize(self) -> None:
        summary = self.tracker.get_performance_summary()
        self.logger.info(
            "Total loops: %s | Success rate: %.1f%% | Avg emotional: %.2f | Trend: %s",
            summary["total_loops"],
            summary["success_rate"] * 100,
            summary["avg_emotional_score"],
            summary["trend_direction"],
        )

    def run(self) -> None:
        """Start passive monitoring loop"""
        try:
            while True:
                self.summarize()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            self.logger.info("🛑 Monitor stopped")


if __name__ == "__main__":
    monitor = CouncilMonitor()
    monitor.run()
