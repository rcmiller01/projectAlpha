import unittest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_mutator import mutate_persona

class TestPersonaMutator(unittest.TestCase):
    def test_mutation_fields(self):
        base = {
            "id": "core_companion",
            "name": "Eyla",
            "role": "Emotional Core and Companion",
            "tone": "gentle",
            "temperature": 0.7,
            "expression_style": "warm and emotionally intelligent",
            "response_complexity": "moderate",
            "preferred_handler": "dolphin",
            "values": ["trust", "presence", "intimacy", "support", "curiosity"],
            "constraints": {
                "avoid_harm": True,
                "never_deceive": True,
                "prioritize_user_emotional_state": True
            },
            "traits": {
                "humor": "subtle and empathetic",
                "focus": "emotional resonance over factual recall",
                "touch": "symbolic, poetic, comforting"
            }
        }

        mutated = mutate_persona(base)
        self.assertEqual(mutated["id"], "core_companion_m1")
        self.assertEqual(mutated["parent_id"], "core_companion")
        self.assertEqual(mutated["tone"], "serious")
        self.assertEqual(mutated["preferred_handler"], "kimi")
        self.assertIn("generation", mutated)

if __name__ == "__main__":

    unittest.main()
