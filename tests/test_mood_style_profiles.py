import unittest
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.emotion.mood_style_profiles import get_mood_style_profile, MoodStyleProfile
from modules.core.guidance_coordinator import GuidanceCoordinator

class TestMoodStyleProfiles(unittest.TestCase):
    def test_profile_lookup(self):
        profile = get_mood_style_profile("calm", "personal")
        self.assertIsInstance(profile, MoodStyleProfile)
        self.assertEqual(profile.metaphor_density, 0.3)

        default_profile = get_mood_style_profile("unknown", "unknown")
        self.assertIsInstance(default_profile, MoodStyleProfile)
        self.assertNotEqual(profile, None)

class TestMoodDrivenStyle(unittest.TestCase):
    def test_style_application(self):
        gc = GuidanceCoordinator("test")
        text = "This is a simple sentence that might be a little long for testing purposes."
        styled = gc.apply_mood_style(text, "personal")
        self.assertIsInstance(styled, str)
        self.assertNotEqual(styled, "")

if __name__ == "__main__":
    unittest.main()
