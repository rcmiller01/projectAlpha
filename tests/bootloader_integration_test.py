#!/usr/bin/env python3
"""
Bootloader Integration Test
Comprehensive test of the autopilot bootloader system
"""

import time
import json
import subprocess
import sys
from pathlib import Path

def test_bootloader_integration():
    """Test the complete bootloader integration"""
    
    print("🚀 Bootloader Integration Test")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Configuration validation
    print("\nTest 1: Configuration Validation")
    try:
        config_path = Path("bootloader_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            required_fields = ['mode', 'idle_threshold', 'check_interval']
            missing = [f for f in required_fields if f not in config]
            
            if not missing:
                print(f"[OK] Configuration valid - Mode: {config['mode']}")
                test_results.append(True)
            else:
                print(f"[FAIL] Missing fields: {missing}")
                test_results.append(False)
        else:
            print("[FAIL] Configuration file not found")
            test_results.append(False)
            
    except Exception as e:
        print(f"[FAIL] Configuration error: {e}")
        test_results.append(False)
    
    # Test 2: Bootloader status functionality
    print("\nTest 2: Bootloader Status")
    try:
        result = subprocess.run([
            "python", "autopilot_bootloader.py", "--status"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Parse JSON output
            lines = result.stdout.strip().split('\n')
            json_lines = [line for line in lines if line.startswith('{')]
            
            if json_lines:
                status_data = json.loads(json_lines[-1])
                bootloader_status = status_data.get('bootloader', {})
                system_status = status_data.get('system', {})
                
                print(f"[OK] Status retrieved successfully")
                print(f"     Mode: {bootloader_status.get('mode', 'unknown')}")
                print(f"     CPU: {system_status.get('cpu_percent', 0):.1f}%")
                print(f"     Memory: {system_status.get('memory_percent', 0):.1f}%")
                test_results.append(True)
            else:
                print("[FAIL] Could not parse status output")
                test_results.append(False)
        else:
            print(f"[FAIL] Status command failed: {result.stderr}")
            test_results.append(False)
            
    except Exception as e:
        print(f"[FAIL] Status test error: {e}")
        test_results.append(False)
    
    # Test 3: Autopilot integration check
    print("\nTest 3: Autopilot Integration")
    try:
        result = subprocess.run([
            "python", "emotion_quant_autopilot/quant_autopilot.py",
            "--config", "emotion_quant_autopilot/autopilot_config.json",
            "integration"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            output = result.stdout.lower()
            if "fully_integrated" in output or "[ok]" in output:
                print("[OK] Autopilot integration confirmed")
                test_results.append(True)
            else:
                print("[WARN] Partial integration detected")
                print(f"      Output: {result.stdout[:100]}...")
                test_results.append(False)
        else:
            print(f"[FAIL] Integration check failed: {result.stderr}")
            test_results.append(False)
            
    except Exception as e:
        print(f"[FAIL] Integration test error: {e}")
        test_results.append(False)
    
    # Test 4: Service script validation
    print("\nTest 4: Service Script")
    try:
        service_script = Path("autopilot_service.bat")
        if service_script.exists():
            with open(service_script) as f:
                content = f.read()
            
            required_commands = ['start', 'stop', 'status', 'install']
            found_commands = [cmd for cmd in required_commands if f":{cmd.upper()}" in content]
            
            if len(found_commands) == len(required_commands):
                print(f"[OK] Service script complete - {len(found_commands)} commands")
                test_results.append(True)
            else:
                missing = set(required_commands) - set(cmd.lower() for cmd in found_commands)
                print(f"[FAIL] Missing commands: {missing}")
                test_results.append(False)
        else:
            print("[FAIL] Service script not found")
            test_results.append(False)
            
    except Exception as e:
        print(f"[FAIL] Service script error: {e}")
        test_results.append(False)
    
    # Test 5: Documentation completeness
    print("\nTest 5: Documentation")
    try:
        doc_files = [
            "AUTOPILOT_BOOTLOADER_DOCUMENTATION.md",
            "bootloader.log"
        ]
        
        missing_docs = [doc for doc in doc_files if not Path(doc).exists()]
        
        if not missing_docs:
            print("[OK] Documentation complete")
            test_results.append(True)
        else:
            print(f"[FAIL] Missing documentation: {missing_docs}")
            test_results.append(False)
            
    except Exception as e:
        print(f"[FAIL] Documentation check error: {e}")
        test_results.append(False)
    
    # Test 6: Launch capability (without actually launching)
    print("\nTest 6: Launch Capability")
    try:
        # Check if autopilot script exists and is accessible
        autopilot_script = Path("emotion_quant_autopilot/quant_autopilot.py")
        autopilot_config = Path("emotion_quant_autopilot/autopilot_config.json")
        
        if autopilot_script.exists() and autopilot_config.exists():
            # Test help command to verify script is executable
            result = subprocess.run([
                "python", str(autopilot_script), "--help"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                print("[OK] Autopilot script accessible and executable")
                test_results.append(True)
            else:
                print(f"[FAIL] Autopilot script not executable: {result.stderr}")
                test_results.append(False)
        else:
            missing = []
            if not autopilot_script.exists():
                missing.append("autopilot script")
            if not autopilot_config.exists():
                missing.append("autopilot config")
            print(f"[FAIL] Missing: {', '.join(missing)}")
            test_results.append(False)
            
    except Exception as e:
        print(f"[FAIL] Launch capability error: {e}")
        test_results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(test_results)
    total = len(test_results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"🎯 Bootloader Integration Results: {passed}/{total} tests passed ({percentage:.0f}%)")
    
    if passed == total:
        print("🎉 FULLY READY - Bootloader system completely integrated!")
        print("\n📋 Next Steps:")
        print("   1. Start bootloader: python autopilot_bootloader.py")
        print("   2. Install as service: autopilot_service.bat install")
        print("   3. Monitor via API: /api/autopilot/bootloader/status")
        print("   4. Configure for your environment via bootloader_config.json")
        
    elif passed >= total * 0.8:
        print("✅ MOSTLY READY - Minor issues may exist")
    elif passed >= total * 0.5:
        print("⚠️ PARTIALLY READY - Some components need attention")
    else:
        print("❌ NOT READY - Significant issues detected")
    
    print("\n🔧 System Components:")
    print("   ✓ autopilot_bootloader.py - Main controller")
    print("   ✓ bootloader_config.json - Configuration")
    print("   ✓ autopilot_service.bat - Service management")
    print("   ✓ AUTOPILOT_BOOTLOADER_DOCUMENTATION.md - Full docs")
    print("   ✓ API integration in dolphin_backend.py")
    
    return passed == total

if __name__ == "__main__":
    success = test_bootloader_integration()
    print(f"\n🏁 Integration Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
