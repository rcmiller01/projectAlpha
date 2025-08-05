"""
Inner Monologue - Autonomous Emotional Reflection Loop

This module simulates the AI's "inner world," allowing it to reflect
on interactions and evolve its emotional understanding even when idle.
"""

import asyncio
import time
import logging
import sys
import os
from typing import Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.reflection.emotion_reflector import EmotionReflector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InnerMonologue:
    """
    Manages the AI's autonomous reflection loop, simulating an "inner monologue."
    """

    def __init__(self, idle_threshold_seconds: int = 300):
        """
        Initializes the InnerMonologue.

        Args:
            idle_threshold_seconds (int): Time in seconds of user inactivity to trigger reflection.
        """
        self.emotion_reflector = EmotionReflector()
        self.idle_threshold_seconds = idle_threshold_seconds
        self.last_user_interaction_time: float = time.time()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def record_user_interaction(self):
        """
        Records the timestamp of the latest user interaction.
        This should be called from the main application loop whenever the user acts.
        """
        self.last_user_interaction_time = time.time()
        logger.debug("User interaction recorded.")

    async def _reflection_loop(self):
        """
        The main asynchronous loop that checks for idle time and triggers reflection.
        """
        while self._running:
            idle_time = time.time() - self.last_user_interaction_time

            if idle_time > self.idle_threshold_seconds:
                logger.info(f"User has been idle for {idle_time:.0f} seconds. Triggering inner reflection.")
                
                # Trigger a daily reflection cycle
                try:
                    summary = self.emotion_reflector.run_daily_reflection()
                    logger.info(f"Inner reflection complete. Dominant emotion: {summary.dominant_emotion}")
                    # Reset timer after reflection to avoid continuous looping
                    self.record_user_interaction() 
                except Exception as e:
                    logger.error(f"An error occurred during inner reflection: {e}")

            await asyncio.sleep(60)  # Check for idle state every minute

    def start(self):
        """
        Starts the inner monologue loop in the background.
        """
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._reflection_loop())
            logger.info("Inner monologue has started.")

    def stop(self):
        """
        Stops the inner monologue loop.
        """
        if self._running and self._task:
            self._running = False
            self._task.cancel()
            logger.info("Inner monologue has been stopped.")

# Example Usage
async def main():
    """
    Example of how to run the InnerMonologue module.
    """
    print("Starting the Inner Monologue simulation.")
    monologue = InnerMonologue(idle_threshold_seconds=10) # Shortened for demo
    monologue.start()

    try:
        # Simulate user interactions
        print("Simulating user activity for 30 seconds...")
        for i in range(3):
            await asyncio.sleep(5)
            monologue.record_user_interaction()
            print(f"[{time.time():.0f}] User interaction occurred.")
        
        print("\nSimulating user going idle. Reflection should trigger in 10 seconds.")
        await asyncio.sleep(35) # Wait for reflection to run and then some

    except asyncio.CancelledError:
        pass
    finally:
        monologue.stop()
        print("\nInner Monologue simulation finished.")


if __name__ == "__main__":
    # To run this example, you might need to ensure the data directories exist
    if not os.path.exists("data/reflection"):
        os.makedirs("data/reflection")
        
    asyncio.run(main())
