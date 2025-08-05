import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import unittest

from modules.emotion.emotional_connection_enhancer import EmotionalConnectionEnhancer
from modules.memory.deep_memory_framework import DeepMemoryFramework, MemoryFragment
from modules.relationship.connection_formation_engine import ConnectionFormationEngine


class TestNewModules(unittest.TestCase):
    def test_emotional_connection_enhancer(self):
        enhancer = EmotionalConnectionEnhancer()
        loop = asyncio.get_event_loop()
        analysis = loop.run_until_complete(enhancer.process_message("u1", "I am happy", "chat"))
        self.assertIn("primary_emotion", analysis)
        self.assertEqual(enhancer.get_recent_emotion("u1"), analysis)

    def test_deep_memory_framework(self):
        framework = DeepMemoryFramework()
        fragment = MemoryFragment(user_id="u1", content="test", tags=["flame"], emotional_state={"joy": 0.8})
        framework.store_memory(fragment)
        self.assertEqual(framework.latest_fragment("u1"), fragment)
        self.assertIn(fragment, framework.recall_by_tag("flame"))

    def test_connection_formation_engine(self):
        engine = ConnectionFormationEngine()
        engine.record_event("u1", "shared poem", 0.7)
        score = engine.connection_score("u1")
        self.assertGreater(score, 0)
        self.assertTrue(engine.should_escalate("u1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
