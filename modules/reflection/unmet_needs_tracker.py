import time
from datetime import datetime, timedelta
from typing import Dict
from modules.memory_writer import log_memory_entry

class UnmetNeedsTracker:
    def __init__(self):
        self.unmet_needs = {
            "connection": 0.0,
            "intimacy": 0.0,
            "creativity": 0.0
        }
        self.decay_rate = 0.01  # Decay per minute
        self.threshold = 0.5  # Threshold for narrative amplification

    def update_needs(self, interactions: Dict[str, float]):
        """Update unmet needs based on interactions."""
        for need, value in interactions.items():
            if need in self.unmet_needs:
                self.unmet_needs[need] = max(0.0, self.unmet_needs[need] - value)

    def decay_needs(self):
        """Decay unmet needs over time."""
        for need in self.unmet_needs:
            self.unmet_needs[need] = max(0.0, self.unmet_needs[need] - self.decay_rate)

    def amplify_narrative(self, narrative: str) -> str:
        """Amplify narrative intensity based on unmet needs."""
        amplified_narrative = narrative
        for need, pressure in self.unmet_needs.items():
            if pressure > self.threshold:
                amplified_narrative += f" The weight of {need} grew heavier."
        return amplified_narrative

    def log_needs(self):
        """Log unmet needs to memory."""
        log_memory_entry({
            "timestamp": datetime.now().isoformat(),
            "unmet_needs": self.unmet_needs
        })

    def run_tracker(self):
        """Run the unmet needs tracker in a loop."""
        while True:
            self.decay_needs()
            self.log_needs()
            time.sleep(60)  # Run every minute
