# test_phase3.py
# Phase 3: Backend Completion & WebUI Integration Tests

import unittest
import json
import time
from typing import Dict, Any

# Import Phase 3 components
from backend.api.utils.llm_router import llm_router
from backend.api.engines.mia_engine import mia_engine
from backend.api.engines.solene_engine import solene_engine
from backend.api.engines.lyra_engine import lyra_engine
from backend.api.engines.doc_engine import doc_engine
from modules.character.consistent_character_generator import consistent_character_generator
from modules.ui.ui_mode_manager import ui_mode_manager, UIMode, InterfaceType
from modules.animation.avatar_animation_system import avatar_animation_system, AnimationMethod, AnimationType

class Phase3BackendTests(unittest.TestCase):
    """Test Phase 3 backend completion features"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_messages = {
            "romantic": "I love spending time with you",
            "technical": "How do I implement a binary search tree in Python?",
            "mystical": "I wonder about the mysteries of the universe",
            "sophisticated": "What are your thoughts on existential philosophy?"
        }
    
    def test_llm_router_initialization(self):
        """Test LLM router initialization"""
        print("\n🧪 Testing LLM Router Initialization...")
        
        # Test available models
        models = llm_router.get_available_models()
        expected_models = ["mythomax", "openchat", "qwen2", "kimik2"]
        
        for model in expected_models:
            self.assertIn(model, models, f"Model {model} should be available")
        
        print(f"✅ Available models: {models}")
        
        # Test model info
        for model in expected_models:
            info = llm_router.get_model_info(model)
            self.assertIn("model", info)
            self.assertIn("name", info)
            self.assertIn("endpoint", info)
            self.assertTrue(info["available"])
            print(f"✅ Model {model} info: {info['name']}")
    
    def test_llm_router_caching(self):
        """Test LLM router caching functionality"""
        print("\n🧪 Testing LLM Router Caching...")
        
        # Test cache stats
        stats = llm_router.get_cache_stats()
        self.assertIn("total_entries", stats)
        self.assertIn("active_entries", stats)
        self.assertIn("cache_ttl", stats)
        print(f"✅ Cache stats: {stats}")
        
        # Test cache clearing
        llm_router.clear_cache()
        stats_after_clear = llm_router.get_cache_stats()
        self.assertEqual(stats_after_clear["total_entries"], 0)
        print("✅ Cache cleared successfully")
    
    def test_mia_engine(self):
        """Test Mia engine functionality"""
        print("\n🧪 Testing Mia Engine...")
        
        # Test mood analysis
        mood = mia_engine.analyze_mia_mood("I'm so happy to see you!")
        self.assertEqual(mood, "happy")
        print(f"✅ Mia mood analysis: {mood}")
        
        # Test romantic gesture
        gesture = mia_engine.get_romantic_gesture("romantic")
        self.assertIsInstance(gesture, str)
        self.assertGreater(len(gesture), 0)
        print(f"✅ Mia gesture: {gesture}")
        
        # Test response generation (with fallback)
        result = mia_engine.handle_mia("Hello Mia!")
        self.assertIn("success", result)
        self.assertIn("response", result)
        self.assertIn("persona", result)
        self.assertEqual(result["persona"], "mia")
        print(f"✅ Mia response: {result['response'][:50]}...")
    
    def test_solene_engine(self):
        """Test Solene engine functionality"""
        print("\n🧪 Testing Solene Engine...")
        
        # Test mood analysis
        mood = solene_engine.analyze_solene_mood("What are your thoughts on philosophy?")
        self.assertEqual(mood, "intellectual")
        print(f"✅ Solene mood analysis: {mood}")
        
        # Test sophisticated gesture
        gesture = solene_engine.get_sophisticated_gesture("mysterious")
        self.assertIsInstance(gesture, str)
        self.assertGreater(len(gesture), 0)
        print(f"✅ Solene gesture: {gesture}")
        
        # Test response generation (with fallback)
        result = solene_engine.handle_solene("Hello Solene!")
        self.assertIn("success", result)
        self.assertIn("response", result)
        self.assertIn("persona", result)
        self.assertEqual(result["persona"], "solene")
        print(f"✅ Solene response: {result['response'][:50]}...")
    
    def test_lyra_engine(self):
        """Test Lyra engine functionality"""
        print("\n🧪 Testing Lyra Engine...")
        
        # Test mood analysis
        mood = lyra_engine.analyze_lyra_mood("I wonder about the mysteries of life")
        self.assertEqual(mood, "curious")
        print(f"✅ Lyra mood analysis: {mood}")
        
        # Test symbol detection
        symbols = lyra_engine.detect_lyra_symbols("I see my reflection in the mirror")
        self.assertIn("mirror", symbols)
        print(f"✅ Lyra symbols detected: {symbols}")
        
        # Test response generation (with fallback)
        result = lyra_engine.handle_lyra("Hello Lyra!")
        self.assertIn("success", result)
        self.assertIn("response", result)
        self.assertIn("persona", result)
        self.assertEqual(result["persona"], "lyra")
        print(f"✅ Lyra response: {result['response'][:50]}...")
    
    def test_doc_engine(self):
        """Test Doc engine functionality"""
        print("\n🧪 Testing Doc Engine...")
        
        # Test technical context analysis
        context = doc_engine.analyze_technical_context("How do I implement a Python function?")
        self.assertTrue(context["is_technical"])
        self.assertIn("python", context["languages"])
        print(f"✅ Doc technical analysis: {context}")
        
        # Test coding relevance
        is_coding = doc_engine.is_coding_related("function class method variable")
        self.assertTrue(is_coding)
        print(f"✅ Doc coding relevance: {is_coding}")
        
        # Test suggestions
        suggestions = doc_engine.get_specialization_suggestions(context)
        self.assertIsInstance(suggestions, list)
        print(f"✅ Doc suggestions: {suggestions}")
        
        # Test response generation (with fallback)
        result = doc_engine.handle_doc("How do I debug this code?")
        self.assertIn("success", result)
        self.assertIn("response", result)
        self.assertIn("persona", result)
        self.assertEqual(result["persona"], "doc")
        print(f"✅ Doc response: {result['response'][:50]}...")
    
    def test_character_generation_integration(self):
        """Test character generation with all personas"""
        print("\n🧪 Testing Character Generation Integration...")
        
        personas = ["mia", "solene", "lyra", "doc"]
        
        for persona in personas:
            # Test character initialization
            result = consistent_character_generator.initialize_character(persona)
            self.assertIn("success", result)
            self.assertEqual(result["persona_id"], persona)
            print(f"✅ Character initialized for {persona}")
            
            # Test character generation
            gen_result = consistent_character_generator.generate_character(
                persona_id=persona,
                aspect="full",
                mood="neutral"
            )
            self.assertIn("success", gen_result)
            self.assertIn("image_url", gen_result)
            print(f"✅ Character generated for {persona}")
    
    def test_ui_mode_management(self):
        """Test UI mode management functionality"""
        print("\n🧪 Testing UI Mode Management...")
        
        # Test mode switching
        ui_mode_manager.switch_mode(UIMode.COMPANION, InterfaceType.WEB)
        self.assertEqual(ui_mode_manager.current_mode, UIMode.COMPANION)
        self.assertTrue(ui_mode_manager.is_avatar_visible())
        print("✅ Switched to Companion mode")
        
        ui_mode_manager.switch_mode(UIMode.DEV, InterfaceType.WEB)
        self.assertEqual(ui_mode_manager.current_mode, UIMode.DEV)
        self.assertFalse(ui_mode_manager.is_avatar_visible())
        print("✅ Switched to Dev mode")
        
        # Test persona configurations
        personas = ["mia", "solene", "lyra", "doc"]
        for persona in personas:
            config = ui_mode_manager.get_persona_config(persona)
            self.assertIsInstance(config, dict)
            print(f"✅ {persona} config: {config}")
    
    def test_animation_system_integration(self):
        """Test animation system integration"""
        print("\n🧪 Testing Animation System Integration...")
        
        # Test available animations
        animations = avatar_animation_system.get_available_animations()
        self.assertIsInstance(animations, dict)
        self.assertIn("real_time_generation", animations)
        print(f"✅ Available animations: {list(animations.keys())}")
        
        # Test animation method info
        for method_name in animations.keys():
            try:
                method = AnimationMethod(method_name)
                info = avatar_animation_system.get_animation_method_info(method)
                self.assertIn("method", info)
                print(f"✅ Animation method {method_name} info retrieved")
            except ValueError:
                print(f"⚠️  Animation method {method_name} not in enum")
        
        # Test parametric animation creation
        result = avatar_animation_system.create_parametric_animation(
            AnimationType.EXPRESSION,
            {"expression": "smile", "intensity": 0.8},
            2.0
        )
        self.assertIn("success", result)
        print("✅ Parametric animation created")

class Phase3IntegrationTests(unittest.TestCase):
    """Test Phase 3 integration features"""
    
    def test_persona_switching(self):
        """Test switching between all personas"""
        print("\n🧪 Testing Persona Switching...")
        
        personas = [
            ("mia", "mythomax", "romantic_companion"),
            ("solene", "openchat", "romantic_companion"),
            ("lyra", "qwen2", "mystical_entity"),
            ("doc", "kimik2", "coding_assistant")
        ]
        
        for persona_id, expected_model, expected_type in personas:
            # Test persona availability
            available = ui_mode_manager.get_available_personas()
            self.assertIn(persona_id, available)
            
            persona_info = available[persona_id]
            self.assertEqual(persona_info["llm_model"], expected_model)
            self.assertEqual(persona_info["type"], expected_type)
            print(f"✅ {persona_id} persona verified: {expected_model} - {expected_type}")
    
    def test_llm_model_consistency(self):
        """Test LLM model consistency across personas"""
        print("\n🧪 Testing LLM Model Consistency...")
        
        persona_models = {
            "mia": "mythomax",
            "solene": "openchat", 
            "lyra": "qwen2",
            "doc": "kimik2"
        }
        
        for persona, expected_model in persona_models.items():
            # Test engine model assignment
            if persona == "mia":
                engine_model = mia_engine.llm_model
            elif persona == "solene":
                engine_model = solene_engine.llm_model
            elif persona == "lyra":
                engine_model = lyra_engine.llm_model
            elif persona == "doc":
                engine_model = doc_engine.llm_model
            
            self.assertEqual(engine_model, expected_model)
            print(f"✅ {persona} engine uses {engine_model}")
    
    def test_emotional_hooks_consistency(self):
        """Test emotional hooks consistency"""
        print("\n🧪 Testing Emotional Hooks Consistency...")
        
        # Test that romantic companions have emotional hooks
        self.assertTrue(mia_engine.emotional_hooks)
        self.assertTrue(solene_engine.emotional_hooks)
        self.assertTrue(lyra_engine.emotional_hooks)
        
        # Test that coding assistant doesn't have emotional hooks
        self.assertFalse(doc_engine.emotional_hooks)
        
        print("✅ Emotional hooks consistency verified")
    
    def test_character_template_consistency(self):
        """Test character template consistency"""
        print("\n🧪 Testing Character Template Consistency...")
        
        personas = ["mia", "solene", "lyra", "doc"]
        
        for persona in personas:
            template = consistent_character_generator.persona_templates.get(persona)
            self.assertIsNotNone(template)
            self.assertIn("base_appearance", template)
            self.assertIn("style_preferences", template)
            self.assertIn("personality_traits", template)
            self.assertIn("llm_model", template)
            self.assertIn("persona_type", template)
            
            print(f"✅ {persona} character template verified")

def run_phase3_tests():
    """Run all Phase 3 tests"""
    print("🚀 Starting Phase 3: Backend Completion & WebUI Integration Tests")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add backend tests
    suite.addTest(unittest.makeSuite(Phase3BackendTests))
    suite.addTest(unittest.makeSuite(Phase3IntegrationTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Phase 3 Test Results Summary:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️  Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All Phase 3 tests passed! Backend completion successful.")
    else:
        print("\n🔧 Some tests failed. Please review and fix issues.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_phase3_tests() 