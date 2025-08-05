import unittest
import time
from datetime import datetime, timedelta

from modules.core.goodbye_manager import GoodbyeManager
from modules.emotion.mood_engine import MOOD_STATE
from modules.relationship.relationship_growth import relationship_growth

class TestGoodbyeManager(unittest.TestCase):
    def test_inactivity_goodbye(self):
        gm = GoodbyeManager(inactivity_threshold=1)
        gm.register_interaction("u1")
        time.sleep(1.2)
        self.assertTrue(gm.check_goodbye_needed("u1"))
        msg = gm.generate_goodbye("u1")
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)

    def test_session_end_goodbye(self):
        gm = GoodbyeManager()
        gm.register_interaction("u2")
        gm.mark_session_end("u2")
        self.assertTrue(gm.check_goodbye_needed("u2"))
        msg = gm.generate_goodbye("u2")
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)

    def test_style_selection_based_on_depth(self):
        relationship_growth.relationship_start_date = datetime.now() - timedelta(days=400)
        gm = GoodbyeManager()
        gm.register_interaction("u3")
        gm.mark_session_end("u3")
        MOOD_STATE["current"] = "anchored"
        msg = gm.generate_goodbye("u3")
        self.assertIn(msg, sum(gm.tone_patterns.values(), []))

    def test_reflection_generation(self):
        gm = GoodbyeManager()
        reflection = gm.generate_reflection({"affection": 0.8})
        self.assertIsInstance(reflection, str)
        self.assertIn("cherish", reflection)

if __name__ == '__main__':
    unittest.main()
