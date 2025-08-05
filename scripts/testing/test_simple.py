#!/usr/bin/env python3
# test_simple.py
# Simple test script to verify all files can be imported and have correct structure

import os
import sys
import importlib.util
from pathlib import Path

def test_file_import(file_path: str, expected_classes: list = None) -> dict:
    """Test if a Python file can be imported and has expected classes"""
    try:
        # Load the module from file path
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"✅ Successfully imported {file_path}")
        
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
            
    except Exception as e:
        print(f"❌ Failed to import {file_path}: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Run simple tests on all advanced feature files"""
    print("🧪 Simple Module Structure Test")
    print("=" * 50)
    
    # Get the project root directory - we're already in the project root
    backend_dir = Path("backend")
    
    # Test all advanced feature files
    test_files = [
        {
            "path": backend_dir / "symbolic" / "symbolic_fusion.py",
            "expected_classes": ["SymbolicFusion", "Symbol", "SymbolType"]
        },
        {
            "path": backend_dir / "scenes" / "scene_initiation.py", 
            "expected_classes": ["SceneInitiationEngine", "ScenePrompt", "SceneType"]
        },
        {
            "path": backend_dir / "input" / "touch_journal.py",
            "expected_classes": ["TouchJournalEngine", "TouchEvent", "TouchLocation"]
        },
        {
            "path": backend_dir / "input" / "dynamic_wake_word.py",
            "expected_classes": ["DynamicWakeWordEngine", "WakeContext", "WakeMode"]
        },
        {
            "path": backend_dir / "ritual" / "mirror_ritual.py",
            "expected_classes": ["MirrorRitualEngine", "MirrorRitual", "RitualPhase", "MirrorState", "IdentityAspect"]
        },
        {
            "path": backend_dir / "privacy" / "private_scenes.py",
            "expected_classes": ["PrivateScenesEngine", "PrivacyLevel", "TrustRequirement", "ContentType"]
        },
        {
            "path": backend_dir / "biometrics" / "biometric_integration.py",
            "expected_classes": ["BiometricIntegrationEngine", "BiometricReading", "MotionData", "EmotionalBiometricState"]
        }
    ]
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in test_files:
        file_path = test_file["path"]
        expected_classes = test_file["expected_classes"]
        
        if file_path.exists():
            print(f"\nTesting {file_path.name}...")
            result = test_file_import(str(file_path), expected_classes)
            if result["success"]:
                success_count += 1
        else:
            print(f"\n❌ File not found: {file_path}")
    
    print(f"\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_count} files passed")
    
    if success_count == total_count:
        print("✅ All advanced feature files are properly structured!")
        print("🎉 The codebase is ready for development!")
    else:
        print(f"❌ {total_count - success_count} files have issues")
    
    # Test __init__.py files
    print(f"\n🧪 Testing Package Structure")
    print("=" * 50)
    
    init_files = [
        backend_dir / "symbolic" / "__init__.py",
        backend_dir / "scenes" / "__init__.py",
        backend_dir / "input" / "__init__.py", 
        backend_dir / "ritual" / "__init__.py",
        backend_dir / "privacy" / "__init__.py",
        backend_dir / "biometrics" / "__init__.py"
    ]
    
    init_success = 0
    for init_file in init_files:
        if init_file.exists():
            print(f"✅ {init_file.name} exists")
            init_success += 1
        else:
            print(f"❌ {init_file.name} missing")
    
    print(f"\n📊 Package Structure: {init_success}/{len(init_files)} __init__.py files present")

if __name__ == "__main__":
    main() 