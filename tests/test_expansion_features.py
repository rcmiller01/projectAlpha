import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.emotion.mood_inflection import MoodInflection
from modules.symbolic.symbol_resurrection import SymbolResurrectionManager
from modules.core.goodbye_protocol import GoodbyeProtocol
from modules.relationship.connection_depth_tracker import ConnectionDepthTracker
from modules.memory.narrative_memory_templates import NarrativeMemoryTemplateManager
from datetime import timedelta

class TestExpansionModules(unittest.TestCase):
    def test_mood_inflection(self):
        mi = MoodInflection()
        text = mi.apply_inflection("Hello", mode="creative", mood="upbeat")
        self.assertIn("Hello", text)

    def test_symbol_resurrection(self):
        sr = SymbolResurrectionManager(threshold=timedelta(seconds=0))
        sr.register_usage("user", "garden")
        self.assertIn("garden", sr.check_resurrection("user"))

    def test_goodbye_protocol(self):
        gp = GoodbyeProtocol()
        msg = gp.select_goodbye("tender")
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)

    def test_connection_depth(self):
        cd = ConnectionDepthTracker()
        cd.update_depth("user")
        self.assertFalse(cd.should_initiate_ritual("user", threshold=5))
        cd.update_depth("user", score=5)
        self.assertTrue(cd.should_initiate_ritual("user", threshold=5))

    def test_narrative_templates(self):
        nt = NarrativeMemoryTemplateManager()
        mem = nt.generate_narrative({"symbol": "garden", "event": "walked", "date": "yesterday"})
        self.assertIn("garden", mem)

if __name__ == '__main__':
    unittest.main()
