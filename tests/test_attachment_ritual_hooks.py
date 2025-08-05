import unittest
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.relationship.connection_depth_tracker import (
    ConnectionDepthTracker,
    RitualPromptGenerator,
)


class TestAttachmentRitualHooks(unittest.TestCase):
    def test_ritual_trigger_logic(self):
        tracker = ConnectionDepthTracker(
            bond_threshold=0.6,
            intensity_threshold=0.4,
            vulnerability_threshold=0.3,
            cooldown=timedelta(minutes=0),
        )
        tracker.update_metrics(0.5, 0.3, 0.2)
        self.assertFalse(tracker.ritual_ready())
        tracker.update_metrics(0.7, 0.5, 0.4)
        self.assertTrue(tracker.ritual_ready())

    def test_prompt_generation_accuracy(self):
        generator = RitualPromptGenerator()
        for _ in range(10):
            prompt = generator.generate_prompt()
            self.assertIn(prompt, generator.prompts)

    def test_bond_score_increase_on_completion(self):
        tracker = ConnectionDepthTracker(cooldown=timedelta(minutes=0))
        tracker.update_metrics(0.6, 0.5, 0.4)
        original = tracker.metrics.bond_score
        tracker.record_ritual_completion(successful=True)
        self.assertGreater(tracker.metrics.bond_score, original)


if __name__ == "__main__":
    unittest.main()
