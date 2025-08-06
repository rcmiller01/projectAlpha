#!/usr/bin/env python3
"""Idle Watchdog

Monitors user activity and initiates self-training when the system has been idle
for an extended period. The watchdog checks for idle state every five minutes
and respects the ``IDLE_THRESHOLD_MINUTES`` environment variable.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta

import psutil

from mirror_mode import get_mirror_mode_manager
from trigger_self_train import trigger_self_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IDLE_THRESHOLD_MINUTES = int(os.getenv("IDLE_THRESHOLD_MINUTES", "30"))
CHECK_INTERVAL_SECONDS = 300  # five minutes


class IdleWatchdog:
    """Watchdog that triggers self-training when the user is idle."""

    def __init__(self, idle_minutes: int = IDLE_THRESHOLD_MINUTES):
        self.idle_threshold = timedelta(minutes=idle_minutes)
        self.last_active_time = datetime.now()
        self.running = False

    def mark_active(self) -> None:
        """Record a user activity event."""
        self.last_active_time = datetime.now()
        logger.debug("User activity recorded")

    async def start(self) -> None:
        """Begin monitoring for idle state."""
        self.running = True
        logger.info("Idle watchdog started")
        while self.running:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            await self._check_idle()

    def stop(self) -> None:
        """Stop monitoring for idle state."""
        self.running = False
        logger.info("Idle watchdog stopped")

    async def _check_idle(self) -> None:
        """Check if the system has been idle long enough to trigger training."""
        idle_duration = datetime.now() - self.last_active_time
        cpu_usage = psutil.cpu_percent(interval=None)
        logger.info(
            "Idle check → %.0f seconds idle | CPU %.1f%%",
            idle_duration.total_seconds(),
            cpu_usage,
        )

        if idle_duration >= self.idle_threshold:
            mirror = get_mirror_mode_manager()
            if mirror and getattr(mirror, "is_enabled", False):
                logger.info("Mirror mode active; skipping self-training trigger")
                return

            logger.info("Idle threshold exceeded; initiating self-training")
            await trigger_self_training()


idle_watchdog = IdleWatchdog()

if __name__ == "__main__":
    asyncio.run(idle_watchdog.start())
