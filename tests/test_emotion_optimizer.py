import asyncio
import os
import json
import unittest
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_engine import choose_adjective, choose_emoji
from emotion_optimizer import EmotionOptimizer


class TestEmotionEngine(unittest.TestCase):
    def test_adjective_choice(self):
        adj = choose_adjective(0.8, 0.6)
        self.assertIsInstance(adj, str)
        self.assertNotEqual(adj, "")

    def test_emoji_choice(self):
        emoji = choose_emoji(-0.5, 0.2)
        self.assertEqual(emoji, "😢")


class TestEmotionOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = EmotionOptimizer(template_dir="response_templates")

    def test_optimize_basic(self):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.optimizer.optimize(
                "I had a bad day",
                {"valence": 0.2, "arousal": 0.4},
                {"valence": -0.6, "arousal": 0.7},
                "Companion",
            )
        )
        self.assertIn("response", result)
        self.assertIn("signature", result)
        self.assertIsInstance(result["signature"], dict)


if __name__ == "__main__":
    unittest.main()

