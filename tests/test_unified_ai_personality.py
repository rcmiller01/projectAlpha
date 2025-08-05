"""
Unit tests for the unified AI personality components.
Tests goodbye manager, mood style profiles, and ritual hooks.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

# Import the components we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from goodbye_manager import GoodbyeManager, GoodbyeType
from mood_style_profiles import MoodStyleProfile, get_style_profile, DEFAULT_PROFILES
from ritual_hooks import RitualEngine


class TestGoodbyeManager:
    """Test cases for GoodbyeManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_emotion_manager = Mock()
        self.mock_connection_tracker = Mock()
        self.goodbye_manager = GoodbyeManager(
            self.mock_emotion_manager, 
            self.mock_connection_tracker
        )
    
    def test_determine_goodbye_type_low_bond(self):
        """Test goodbye type determination for low bond depth"""
        result = self.goodbye_manager.determine_goodbye_type("calm", 0.2, 600)
        assert result == GoodbyeType.DIRECT
    
    def test_determine_goodbye_type_high_bond_intimate(self):
        """Test goodbye type determination for high bond depth and intimate mood"""
        result = self.goodbye_manager.determine_goodbye_type("intimate", 0.9, 1200)
        assert result == GoodbyeType.POETIC
    
    def test_determine_goodbye_type_short_session(self):
        """Test goodbye type for short sessions"""
        result = self.goodbye_manager.determine_goodbye_type("calm", 0.6, 200)
        assert result == GoodbyeType.DIRECT
    
    def test_generate_goodbye_with_managers(self):
        """Test goodbye generation with emotion and connection managers"""
        self.mock_emotion_manager.get_current_mood.return_value = "calm"
        self.mock_connection_tracker.get_bond_depth.return_value = 0.8
        
        result = self.goodbye_manager.generate_goodbye(1200)
        
        assert "message" in result
        assert "type" in result
        assert "mood" in result
        assert "bond_depth" in result
        assert "timestamp" in result
        assert result["mood"] == "calm"
        assert result["bond_depth"] == 0.8
        assert isinstance(result["message"], str)
    
    def test_generate_goodbye_without_managers(self):
        """Test goodbye generation without managers (using defaults)"""
        goodbye_manager = GoodbyeManager()
        result = goodbye_manager.generate_goodbye()
        
        assert "message" in result
        assert result["mood"] == "calm"  # default
        assert result["bond_depth"] == 0.5  # default
        assert len(result["message"]) > 0
    
    def test_handle_timeout_goodbye_long_inactive(self):
        """Test timeout goodbye for long inactivity"""
        result = self.goodbye_manager.handle_timeout_goodbye(3600)  # 1 hour
        
        assert result["type"] == GoodbyeType.SILENT
        assert "timeout" in result["reason"]
        assert result["inactive_duration"] == 3600
    
    def test_handle_timeout_goodbye_medium_inactive(self):
        """Test timeout goodbye for medium inactivity"""
        result = self.goodbye_manager.handle_timeout_goodbye(1800)  # 30 minutes
        
        assert result["type"] == GoodbyeType.DIRECT
        assert "timeout" in result["reason"]
    
    def test_handle_timeout_goodbye_short_inactive(self):
        """Test timeout goodbye for short inactivity"""
        result = self.goodbye_manager.handle_timeout_goodbye(900)  # 15 minutes
        
        assert "message" in result
        assert "type" in result
        # Should use normal generation logic for short inactivity
    
    def test_handle_shutdown(self):
        """Test shutdown handling"""
        self.mock_emotion_manager.get_current_mood.return_value = "calm"
        self.mock_connection_tracker.get_bond_depth.return_value = 0.7
        
        with patch('builtins.print') as mock_print:
            result = self.goodbye_manager.handle_shutdown(600)
            
            assert "message" in result
            assert "type" in result
            mock_print.assert_called_once()


class TestMoodStyleProfiles:
    """Test cases for MoodStyleProfile and related functions"""
    
    def test_mood_style_profile_creation(self):
        """Test creating a MoodStyleProfile"""
        profile = MoodStyleProfile(0.5, 10, 0.8, 0.6)
        
        assert profile.metaphor_density == 0.5
        assert profile.avg_sentence_length == 10
        assert profile.warmth_level == 0.8
        assert profile.directness_level == 0.6
    
    def test_get_style_profile_known_combination(self):
        """Test getting a known style profile"""
        profile = get_style_profile("calm", "personal")
        
        assert isinstance(profile, MoodStyleProfile)
        assert profile.metaphor_density == DEFAULT_PROFILES["calm_personal"].metaphor_density
        assert profile.avg_sentence_length == DEFAULT_PROFILES["calm_personal"].avg_sentence_length
    
    def test_get_style_profile_unknown_combination(self):
        """Test getting a style profile for unknown combination (should return default)"""
        profile = get_style_profile("unknown_mood", "unknown_mode")
        
        assert isinstance(profile, MoodStyleProfile)
        assert profile.metaphor_density == 0.5  # default value
        assert profile.avg_sentence_length == 10  # default value
        assert profile.warmth_level == 0.7  # default value
        assert profile.directness_level == 0.7  # default value
    
    def test_default_profiles_exist(self):
        """Test that expected default profiles exist"""
        expected_profiles = ["calm_personal", "anxious_crisis", "intimate_hybrid"]
        
        for profile_name in expected_profiles:
            assert profile_name in DEFAULT_PROFILES
            profile = DEFAULT_PROFILES[profile_name]
            assert isinstance(profile, MoodStyleProfile)
            assert 0 <= profile.metaphor_density <= 1
            assert profile.avg_sentence_length > 0
            assert 0 <= profile.warmth_level <= 1
            assert 0 <= profile.directness_level <= 1
    
    def test_profile_values_in_valid_ranges(self):
        """Test that all default profile values are in valid ranges"""
        for profile_name, profile in DEFAULT_PROFILES.items():
            assert 0 <= profile.metaphor_density <= 1, f"Invalid metaphor_density in {profile_name}"
            assert profile.avg_sentence_length > 0, f"Invalid avg_sentence_length in {profile_name}"
            assert 0 <= profile.warmth_level <= 1, f"Invalid warmth_level in {profile_name}"
            assert 0 <= profile.directness_level <= 1, f"Invalid directness_level in {profile_name}"


class TestRitualEngine:
    """Test cases for RitualEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_connection_tracker = Mock()
        self.ritual_engine = RitualEngine(self.mock_connection_tracker)
    
    def test_ritual_engine_initialization(self):
        """Test RitualEngine initialization"""
        assert self.ritual_engine.connection_depth_tracker == self.mock_connection_tracker
        assert len(self.ritual_engine.bonding_prompts) > 0
        assert all(isinstance(prompt, str) for prompt in self.ritual_engine.bonding_prompts)
    
    def test_check_readiness_high_bond(self):
        """Test readiness check with high bond depth"""
        self.mock_connection_tracker.get_bond_depth.return_value = 0.8
        
        result = self.ritual_engine.check_readiness()
        
        assert result is True
        self.mock_connection_tracker.get_bond_depth.assert_called_once()
    
    def test_check_readiness_low_bond(self):
        """Test readiness check with low bond depth"""
        self.mock_connection_tracker.get_bond_depth.return_value = 0.5
        
        result = self.ritual_engine.check_readiness()
        
        assert result is False
        self.mock_connection_tracker.get_bond_depth.assert_called_once()
    
    def test_get_bonding_prompt_ready(self):
        """Test getting bonding prompt when ready"""
        self.mock_connection_tracker.get_bond_depth.return_value = 0.8
        
        result = self.ritual_engine.get_bonding_prompt()
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result in self.ritual_engine.bonding_prompts
    
    def test_get_bonding_prompt_not_ready(self):
        """Test getting bonding prompt when not ready"""
        self.mock_connection_tracker.get_bond_depth.return_value = 0.5
        
        result = self.ritual_engine.get_bonding_prompt()
        
        assert result == ""
    
    def test_bonding_prompts_quality(self):
        """Test that bonding prompts are meaningful"""
        prompts = self.ritual_engine.bonding_prompts
        
        assert len(prompts) >= 3  # Should have multiple options
        for prompt in prompts:
            assert len(prompt) > 10  # Should be meaningful sentences
            assert prompt.endswith('?')  # Should be questions


class TestIntegration:
    """Integration tests for the unified AI personality components"""
    
    def test_components_work_together(self):
        """Test that all components can work together"""
        # Create mock managers
        emotion_manager = Mock()
        connection_tracker = Mock()
        
        # Set up mock returns
        emotion_manager.get_current_mood.return_value = "calm"
        connection_tracker.get_bond_depth.return_value = 0.8
        
        # Create all components
        goodbye_manager = GoodbyeManager(emotion_manager, connection_tracker)
        ritual_engine = RitualEngine(connection_tracker)
        
        # Test that they can all operate
        goodbye_result = goodbye_manager.generate_goodbye()
        ritual_ready = ritual_engine.check_readiness()
        style_profile = get_style_profile("calm", "personal")
        
        # Verify results
        assert "message" in goodbye_result
        assert ritual_ready is True
        assert isinstance(style_profile, MoodStyleProfile)
    
    def test_mood_consistency_across_components(self):
        """Test that mood handling is consistent across components"""
        # Test that the same mood values work across all components
        test_moods = ["calm", "anxious", "intimate", "excited"]
        test_modes = ["personal", "crisis", "hybrid"]
        
        for mood in test_moods:
            for mode in test_modes:
                # Should not raise exceptions
                style_profile = get_style_profile(mood, mode)
                assert isinstance(style_profile, MoodStyleProfile)
    
    def test_adaptive_behavior_simulation(self):
        """Test simulated adaptive behavior across time"""
        emotion_manager = Mock()
        connection_tracker = Mock()
        
        # Simulate bond growing over time
        bond_levels = [0.3, 0.5, 0.7, 0.9]
        moods = ["neutral", "calm", "engaged", "intimate"]
        
        goodbye_manager = GoodbyeManager(emotion_manager, connection_tracker)
        ritual_engine = RitualEngine(connection_tracker)
        
        for i, (bond, mood) in enumerate(zip(bond_levels, moods)):
            emotion_manager.get_current_mood.return_value = mood
            connection_tracker.get_bond_depth.return_value = bond
            
            goodbye = goodbye_manager.generate_goodbye()
            ritual_ready = ritual_engine.check_readiness()
            
            # As bond increases, should see more sophisticated goodbyes
            assert "message" in goodbye
            assert goodbye["bond_depth"] == bond
            
            # Ritual readiness should increase with bond depth
            if bond > 0.7:
                assert ritual_ready is True
            else:
                assert ritual_ready is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
