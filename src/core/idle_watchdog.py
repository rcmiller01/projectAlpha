#!/usr/bin/env python3
"""Idle Watchdog

Monitors user activity and initiates self-training when the system has been idle
for an extended period. The watchdog checks for idle state every five minutes
and respects the ``IDLE_THRESHOLD_MINUTES`` environment variable.

Enhanced with emotion loop pause functionality for complete idle handling.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import psutil

from core.mirror_mode import get_mirror_mode_manager
from trigger_self_train import trigger_self_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IDLE_THRESHOLD_MINUTES = int(os.getenv("IDLE_THRESHOLD_MINUTES", "30"))
CHECK_INTERVAL_SECONDS = 300  # five minutes


class IdleWatchdog:
    """Watchdog that triggers self-training when the user is idle and manages emotion loop pausing."""

    def __init__(self, idle_minutes: int = IDLE_THRESHOLD_MINUTES):
        self.idle_threshold = timedelta(minutes=idle_minutes)
        self.last_active_time = datetime.now()
        self.running = False
        
        # Emotion loop pause management
        self.emotion_loop_paused = False
        self.emotion_loop_pause_time: Optional[datetime] = None
        self.emotion_loop_instance: Optional[Any] = None
        self.pause_emotion_on_idle = True  # Configuration flag
        
        logger.info(f"IdleWatchdog initialized with {idle_minutes}min threshold and emotion loop pause enabled")

    def mark_active(self) -> None:
        """Record a user activity event and resume emotion loop if paused."""
        self.last_active_time = datetime.now()
        logger.debug("User activity recorded")
        
        # Resume emotion loop if it was paused due to idle
        if self.emotion_loop_paused:
            asyncio.create_task(self._resume_emotion_loop())

    def set_emotion_loop_instance(self, emotion_loop) -> None:
        """Set the emotion loop instance to manage during idle periods."""
        self.emotion_loop_instance = emotion_loop
        logger.debug("Emotion loop instance registered with idle watchdog")

    async def _pause_emotion_loop(self) -> bool:
        """
        Pause the emotion loop during idle periods to conserve resources.
        
        Returns:
            True if successfully paused, False otherwise
        """
        if not self.pause_emotion_on_idle:
            return False
        
        if self.emotion_loop_paused:
            return True  # Already paused
        
        try:
            # Try to pause the emotion loop instance if available
            if self.emotion_loop_instance and hasattr(self.emotion_loop_instance, 'pause'):
                await self.emotion_loop_instance.pause()
                self.emotion_loop_paused = True
                self.emotion_loop_pause_time = datetime.now()
                logger.info("Emotion loop paused due to idle state")
                return True
            
            # Alternative: Try to import and pause emotion loop directly
            try:
                from core.emotion_loop_core import pause_emotion_loop
                success = await pause_emotion_loop()
                if success:
                    self.emotion_loop_paused = True
                    self.emotion_loop_pause_time = datetime.now()
                    logger.info("Emotion loop paused via direct core pause")
                    return True
            except ImportError:
                logger.debug("Direct emotion loop pause not available")
            
            # Fallback: Use environment variable or global flag
            os.environ['EMOTION_LOOP_PAUSED'] = 'true'
            self.emotion_loop_paused = True
            self.emotion_loop_pause_time = datetime.now()
            logger.info("Emotion loop paused via environment flag")
            return True
            
        except Exception as e:
            logger.error(f"Error pausing emotion loop: {e}")
            return False

    async def _resume_emotion_loop(self) -> bool:
        """
        Resume the emotion loop when activity is detected.
        
        Returns:
            True if successfully resumed, False otherwise
        """
        if not self.emotion_loop_paused:
            return True  # Already running
        
        try:
            # Calculate pause duration for logging
            pause_duration = None
            if self.emotion_loop_pause_time:
                pause_duration = datetime.now() - self.emotion_loop_pause_time
            
            # Try to resume the emotion loop instance if available
            if self.emotion_loop_instance and hasattr(self.emotion_loop_instance, 'resume'):
                await self.emotion_loop_instance.resume()
                self.emotion_loop_paused = False
                self.emotion_loop_pause_time = None
                logger.info(f"Emotion loop resumed after {pause_duration} pause")
                return True
            
            # Alternative: Try to resume emotion loop directly
            try:
                from core.emotion_loop_core import resume_emotion_loop
                success = await resume_emotion_loop()
                if success:
                    self.emotion_loop_paused = False
                    self.emotion_loop_pause_time = None
                    logger.info(f"Emotion loop resumed via direct core resume after {pause_duration}")
                    return True
            except ImportError:
                logger.debug("Direct emotion loop resume not available")
            
            # Fallback: Clear environment variable
            if 'EMOTION_LOOP_PAUSED' in os.environ:
                del os.environ['EMOTION_LOOP_PAUSED']
            
            self.emotion_loop_paused = False
            self.emotion_loop_pause_time = None
            logger.info(f"Emotion loop resumed via environment flag after {pause_duration}")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming emotion loop: {e}")
            return False

    def get_emotion_loop_status(self) -> dict:
        """Get current emotion loop pause status."""
        status = {
            "emotion_loop_paused": self.emotion_loop_paused,
            "pause_enabled": self.pause_emotion_on_idle,
            "pause_time": self.emotion_loop_pause_time.isoformat() if self.emotion_loop_pause_time else None,
            "pause_duration_seconds": None,
            "has_emotion_loop_instance": self.emotion_loop_instance is not None
        }
        
        if self.emotion_loop_paused and self.emotion_loop_pause_time:
            duration = datetime.now() - self.emotion_loop_pause_time
            status["pause_duration_seconds"] = duration.total_seconds()
        
        return status

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
        """Check if the system has been idle long enough to trigger training and emotion loop pause."""
        idle_duration = datetime.now() - self.last_active_time
        cpu_usage = psutil.cpu_percent(interval=None)
        logger.info(
            "Idle check → %.0f seconds idle | CPU %.1f%% | Emotion loop paused: %s",
            idle_duration.total_seconds(),
            cpu_usage,
            self.emotion_loop_paused
        )

        # Check if we should pause emotion loop during idle period
        if idle_duration >= self.idle_threshold and not self.emotion_loop_paused:
            logger.info("Idle threshold exceeded; pausing emotion loop")
            await self._pause_emotion_loop()

        # Continue with original idle threshold check for self-training
        if idle_duration >= self.idle_threshold:
            mirror = get_mirror_mode_manager()
            if mirror and getattr(mirror, "is_enabled", False):
                logger.info("Mirror mode active; skipping self-training trigger")
                return

            logger.info("Idle threshold exceeded; initiating self-training")
            await trigger_self_training()
    
    def get_status(self) -> dict:
        """Get comprehensive idle watchdog status including emotion loop state."""
        idle_duration = datetime.now() - self.last_active_time
        
        status = {
            "running": self.running,
            "idle_threshold_minutes": self.idle_threshold.total_seconds() / 60,
            "current_idle_seconds": idle_duration.total_seconds(),
            "is_idle": idle_duration >= self.idle_threshold,
            "last_active_time": self.last_active_time.isoformat(),
            "check_interval_seconds": CHECK_INTERVAL_SECONDS
        }
        
        # Add emotion loop status
        emotion_status = self.get_emotion_loop_status()
        status.update(emotion_status)
        
        return status


idle_watchdog = IdleWatchdog()

if __name__ == "__main__":
    asyncio.run(idle_watchdog.start())
