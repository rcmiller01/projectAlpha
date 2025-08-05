import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.emotion.sensual_expression_alignment import (
    ExpressionDial,
    AnchorSystem,
    SensualExpressionAlignmentSystem,
)


class TestExpressionDial(unittest.TestCase):
    def test_set_and_get_level(self):
        dial = ExpressionDial()
        dial.set_level("desire", 5)
        self.assertEqual(dial.get_level("desire"), 5)
        dial.set_level("desire", 12)
        self.assertEqual(dial.get_level("desire"), dial.max_level)


class TestAnchorSystem(unittest.TestCase):
    def test_enforce_safety(self):
        dial = ExpressionDial()
        dial.set_level("passion", 10)
        anchor = AnchorSystem(dial)
        anchor.enforce_safety()
        self.assertLessEqual(dial.get_level("passion"), anchor.safety_threshold)


class TestSensualSystem(unittest.TestCase):
    def test_feedback_adjustment(self):
        system = SensualExpressionAlignmentSystem("user")
        system.set_emotion_level("affection", 5)
        system.record_user_feedback("affection", 0.5)
        self.assertGreaterEqual(system.dial.get_level("affection"), 5)


if __name__ == "__main__":
    unittest.main()
