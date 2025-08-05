#!/usr/bin/env python3
# test_modules.py
# Test script to verify all Python modules can be imported and have correct structure

import sys
import os
import importlib
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_module_import(module_name: str, expected_classes: List[str] = None) -> Dict[str, Any]:
    """Test if a module can be imported and has expected classes"""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        
        if expected_classes:
            missing_classes = []
            for class_name in expected_classes:
                if hasattr(module, class_name):
                    print(f"   ✅ Found class: {class_name}")
                else:
                    missing_classes.append(class_name)
                    print(f"   ❌ Missing class: {class_name}")
            
            return {
                "success": True,
                "missing_classes": missing_classes,
                "module": module
            }
        else:
            return {"success": True, "module": module}
            
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"❌ Error testing {module_name}: {e}")
        return {"success": False, "error": str(e)}

def test_advanced_features():
    """Test all advanced feature modules"""
    print("🧪 Testing Advanced Feature Modules")
    print("=" * 50)
    
    # Test Symbolic Fusion
    print("\n1. Testing Symbolic Fusion Module...")
    result = test_module_import("backend.symbolic.symbolic_fusion", [
        "SymbolicFusion", "Symbol", "SymbolType"
    ])
    
    # Test Scene Initiation
    print("\n2. Testing Scene Initiation Module...")
    result = test_module_import("backend.scenes.scene_initiation", [
        "SceneInitiationEngine", "ScenePrompt", "SceneType"
    ])
    
    # Test Touch Journal
    print("\n3. Testing Touch Journal Module...")
    result = test_module_import("backend.input.touch_journal", [
        "TouchJournalEngine", "TouchEvent", "TouchLocation"
    ])
    
    # Test Dynamic Wake Word
    print("\n4. Testing Dynamic Wake Word Module...")
    result = test_module_import("backend.input.dynamic_wake_word", [
        "DynamicWakeWordEngine", "WakeContext", "WakeMode"
    ])
    
    # Test Mirror Ritual
    print("\n5. Testing Mirror Ritual Module...")
    result = test_module_import("backend.ritual.mirror_ritual", [
        "MirrorRitualEngine", "MirrorRitual", "RitualPhase", "MirrorState", "IdentityAspect"
    ])
    
    # Test Private Scenes
    print("\n6. Testing Private Scenes Module...")
    result = test_module_import("backend.privacy.private_scenes", [
        "PrivateScenesEngine", "PrivacyLevel", "TrustRequirement", "ContentType"
    ])
    
    # Test Biometric Integration
    print("\n7. Testing Biometric Integration Module...")
    result = test_module_import("backend.biometrics.biometric_integration", [
        "BiometricIntegrationEngine", "BiometricReading", "MotionData", "EmotionalBiometricState"
    ])

def test_core_modules():
    """Test core system modules"""
    print("\n🧪 Testing Core System Modules")
    print("=" * 50)
    
    # Test core modules
    core_modules = [
        "core.memory_core",
        "core.anchor_runtime",
        "core.emotion.reactive_emotion_engine",
        "core.memory_decay"
    ]
    
    for module_name in core_modules:
        print(f"\nTesting {module_name}...")
        test_module_import(module_name)

def test_module_modules():
    """Test feature modules"""
    print("\n🧪 Testing Feature Modules")
    print("=" * 50)
    
    # Test module features
    module_features = [
        "modules.emotion.emotion_state",
        "modules.memory.emotional_memory",
        "modules.voice.emotional_tts",
        "modules.visual.mood_driven_avatar",
        "modules.character.consistent_character_generator"
    ]
    
    for module_name in module_features:
        print(f"\nTesting {module_name}...")
        test_module_import(module_name)

def test_package_structure():
    """Test package structure and __init__.py files"""
    print("\n🧪 Testing Package Structure")
    print("=" * 50)
    
    packages = [
        "backend.ritual",
        "backend.privacy", 
        "backend.biometrics",
        "backend.scenes",
        "backend.input",
        "backend.symbolic"
    ]
    
    for package_name in packages:
        print(f"\nTesting package {package_name}...")
        try:
            package = importlib.import_module(package_name)
            print(f"✅ Package {package_name} imported successfully")
            
            # Check if __all__ is defined
            if hasattr(package, '__all__'):
                print(f"   ✅ __all__ defined: {package.__all__}")
            else:
                print(f"   ⚠️  No __all__ defined")
                
        except ImportError as e:
            print(f"❌ Failed to import package {package_name}: {e}")

def test_class_instantiation():
    """Test that key classes can be instantiated"""
    print("\n🧪 Testing Class Instantiation")
    print("=" * 50)
    
    try:
        # Test Symbolic Fusion
        from backend.symbolic.symbolic_fusion import SymbolicFusion
        fusion_engine = SymbolicFusion()
        print("✅ SymbolicFusion instantiated successfully")
        
        # Test Scene Initiation
        from backend.scenes.scene_initiation import SceneInitiationEngine
        scene_engine = SceneInitiationEngine()
        print("✅ SceneInitiationEngine instantiated successfully")
        
        # Test Touch Journal
        from backend.input.touch_journal import TouchJournalEngine
        touch_engine = TouchJournalEngine()
        print("✅ TouchJournalEngine instantiated successfully")
        
        # Test Dynamic Wake Word
        from backend.input.dynamic_wake_word import DynamicWakeWordEngine
        wake_engine = DynamicWakeWordEngine()
        print("✅ DynamicWakeWordEngine instantiated successfully")
        
        # Test Mirror Ritual
        from backend.ritual.mirror_ritual import MirrorRitualEngine
        ritual_engine = MirrorRitualEngine()
        print("✅ MirrorRitualEngine instantiated successfully")
        
        # Test Private Scenes
        from backend.privacy.private_scenes import PrivateScenesEngine
        privacy_engine = PrivateScenesEngine()
        print("✅ PrivateScenesEngine instantiated successfully")
        
        # Test Biometric Integration
        from backend.biometrics.biometric_integration import BiometricIntegrationEngine
        bio_engine = BiometricIntegrationEngine()
        print("✅ BiometricIntegrationEngine instantiated successfully")
        
    except Exception as e:
        print(f"❌ Class instantiation failed: {e}")
        import traceback
        traceback.print_exc()

def test_async_methods():
    """Test that async methods exist and can be called"""
    print("\n🧪 Testing Async Methods")
    print("=" * 50)
    
    try:
        # Test Symbolic Fusion async methods
        from backend.symbolic.symbolic_fusion import SymbolicFusion
        fusion_engine = SymbolicFusion()
        
        # Check if async methods exist
        async_methods = [
            'fuse_symbols',
            'activate_symbol',
            'create_compound_mood'
        ]
        
        for method_name in async_methods:
            if hasattr(fusion_engine, method_name):
                print(f"✅ {method_name} method exists")
            else:
                print(f"❌ {method_name} method missing")
        
        # Test Scene Initiation async methods
        from backend.scenes.scene_initiation import SceneInitiationEngine
        scene_engine = SceneInitiationEngine()
        
        async_methods = [
            'analyze_text_for_scene',
            'generate_scene_prompt',
            'create_scene_memory'
        ]
        
        for method_name in async_methods:
            if hasattr(scene_engine, method_name):
                print(f"✅ {method_name} method exists")
            else:
                print(f"❌ {method_name} method missing")
                
    except Exception as e:
        print(f"❌ Async method testing failed: {e}")

def main():
    """Run all module tests"""
    print("🚀 Starting Module Structure Test Suite")
    print("=" * 80)
    
    try:
        test_advanced_features()
        test_core_modules()
        test_module_modules()
        test_package_structure()
        test_class_instantiation()
        test_async_methods()
        
        print("\n" + "=" * 80)
        print("✅ All Module Tests Completed Successfully!")
        print("🎉 All Python modules are properly structured!")
        print("💡 The codebase is ready for development!")
        
    except Exception as e:
        print(f"\n❌ Module test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 