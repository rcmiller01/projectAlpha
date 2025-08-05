import os
import json
import unittest

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import personalization_signature as ps

class TestPersonalizationSignature(unittest.TestCase):
    def setUp(self):
        # ensure we start with default file
        if os.path.exists(ps.USER_SIGNATURE_FILE):
            os.remove(ps.USER_SIGNATURE_FILE)

    def test_load_and_save_signature(self):
        sig = ps.load_user_signature()
        self.assertEqual(sig["playfulness"], 0.5)
        sig["playfulness"] = 0.8
        ps.save_user_signature(sig)
        loaded = ps.load_user_signature()
        self.assertEqual(loaded["playfulness"], 0.8)

    def test_update_signature(self):
        dial = {"playfulness": 0.7}
        ps.update_signature(dial, 1)
        sig = ps.load_user_signature()
        self.assertGreaterEqual(sig["playfulness"], 0.5)

    def tearDown(self):
        if os.path.exists(ps.USER_SIGNATURE_FILE):
            os.remove(ps.USER_SIGNATURE_FILE)

if __name__ == "__main__":
    unittest.main()
