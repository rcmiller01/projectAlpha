#!/usr/bin/env python3
"""
Quantization Autopilot Integration Test
Demonstrates the complete emotional quantization autopilot workflow
"""

import time
import json
import subprocess
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_integrated_workflow():
    """Test the complete integrated workflow"""
    
    print("Starting Quantization Integration Test")
    print("=" * 60)
    
    # Test 1: Check integration status
    print("\nTest 1: Integration Status")
    result = subprocess.run([
        "python", "emotion_quant_autopilot/quant_autopilot.py", 
        "--config", "emotion_quant_autopilot/autopilot_config.json", 
        "integration"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("[OK] Integration status check passed")
        print(result.stdout)
    else:
        print("[FAIL] Integration status check failed")
        print(result.stderr)
        return False
    
    # Test 2: Test individual components
    print("\nTest 2: Individual Component Testing")
    
    # Test quantizer
    print("   Testing quantizer...")
    try:
        from quantize_model import quantize_model
        
        result = quantize_model(
            base_model="test-model",
            quantization_method="q4_K_M",
            config={"backend": "mock", "output_dir": "test_output"}
        )
        
        if result.success:
            print(f"   [OK] Quantizer: {result.model_size_mb:.1f}MB in {result.duration_seconds:.1f}s")
        else:
            print(f"   [FAIL] Quantizer failed: {result.error_message}")
            
    except Exception as e:
        print(f"   [FAIL] Quantizer import failed: {e}")
    
    # Test judge
    print("   Testing emotional judge...")
    try:
        from judge_emotion import judge_emotion
        
        result = judge_emotion(
            model_path="test_output/test-model_q4_K_M_20250729_120000.mock",
            base_model="test-model",
            quantization_method="q4_K_M",
            config={"silent_mode": True, "evaluation_count": 5}
        )
        
        if result.success:
            print(f"   [OK] Judge: Score {result.judgment_score:.3f}, Notes: {result.reflection_notes[:50]}...")
        else:
            print(f"   [FAIL] Judge failed: {result.error_message}")
            
    except Exception as e:
        print(f"   [FAIL] Judge import failed: {e}")
    
    # Test database
    print("   Testing enhanced database...")
    try:
        from emotion_core_tracker import EmotionalQuantDatabase
        
        db = EmotionalQuantDatabase("test_integration.db")
        
        # Test logging
        run_id = db.log_autopilot_run(
            run_id="test_integration_run",
            model_path="test/model/path",
            base_model="test-model",
            quantization_method="q4_K_M",
            judgment_score=0.85,
            emotional_deviation=0.05,
            execution_time_minutes=10.0,
            success=True,
            result_summary="Integration test"
        )
        
        print(f"   [OK] Database: Logged run with ID {run_id}")
        
    except Exception as e:
        print(f"   [FAIL] Database test failed: {e}")
    
    # Test state manager
    print("   Testing state manager...")
    try:
        from autopilot_state import AutopilotStateManager
        
        state_mgr = AutopilotStateManager(
            "test_state.json",
            {"enable_watchdog": False}
        )
        
        state_mgr.start_autopilot("test_job", "test_run")
        status = state_mgr.get_status_summary()
        state_mgr.stop_autopilot()
        
        print(f"   [OK] State Manager: Running={status['is_running']}, Jobs={status['jobs_today']}")
        
    except Exception as e:
        print(f"   [FAIL] State manager test failed: {e}")
    
    # Test 3: End-to-end autopilot workflow simulation
    print("\nTest 3: End-to-End Workflow Simulation")
    
    # Add a test job
    result = subprocess.run([
        "python", "emotion_quant_autopilot/quant_autopilot.py",
        "--config", "emotion_quant_autopilot/autopilot_config.json",
        "add-job", "--model", "llama2-test", "--quant", "q6_K", "--priority", "1"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("[OK] Test job added successfully")
        print(f"   {result.stdout.strip()}")
    else:
        print("[FAIL] Failed to add test job")
        print(result.stderr)
    
    # Check queue
    result = subprocess.run([
        "python", "emotion_quant_autopilot/quant_autopilot.py",
        "--config", "emotion_quant_autopilot/autopilot_config.json",
        "queue"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("[OK] Queue status retrieved")
        print(f"   {result.stdout.strip()}")
    else:
        print("[FAIL] Failed to get queue status")
    
    # Test 4: Configuration validation
    print("\nTest 4: Configuration Validation")
    
    config_path = Path("emotion_quant_autopilot/autopilot_config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        required_sections = [
            "idle_monitoring", "safety_limits", "evaluation_settings",
            "output_paths", "notifications", "target_model_size_range_gb"
        ]
        
        missing_sections = [s for s in required_sections if s not in config]
        
        if not missing_sections:
            print("[OK] Configuration is complete")
            print(f"   Target size: {config['target_model_size_range_gb']}GB")
            print(f"   Max daily runs: {config.get('max_active_loops_per_day', 3)}")
            print(f"   Evaluation prompts: {config['evaluation_settings']['evaluation_prompt_count']}")
        else:
            print(f"[FAIL] Missing configuration sections: {missing_sections}")
    else:
        print("[FAIL] Configuration file not found")
    
    # Test 5: Integration readiness check
    print("\nTest 5: Integration Readiness Assessment")
    
    readiness_score = 0
    total_checks = 6
    
    # Check 1: Components available
    try:
        from quantize_model import ModelQuantizer
        from judge_emotion import EmotionalJudge
        from emotion_core_tracker import EmotionalQuantDatabase
        from autopilot_state import AutopilotStateManager
        readiness_score += 1
        print("[OK] All components importable")
    except ImportError as e:
        print(f"[FAIL] Component import issues: {e}")
    
    # Check 2: Database connectivity
    try:
        db = EmotionalQuantDatabase()
        readiness_score += 1
        print("[OK] Database connectivity")
    except Exception as e:
        print(f"[FAIL] Database issues: {e}")
    
    # Check 3: Configuration validity
    if config_path.exists() and not missing_sections:
        readiness_score += 1
        print("[OK] Configuration valid")
    else:
        print("[FAIL] Configuration issues")
    
    # Check 4: Output directories
    try:
        config = json.load(open(config_path))
        output_dirs = config["output_paths"]
        all_dirs_exist = all(Path(path).parent.exists() for path in output_dirs.values())
        if all_dirs_exist:
            readiness_score += 1
            print("[OK] Output directories accessible")
        else:
            print("[FAIL] Some output directories not accessible")
    except Exception:
        print("[FAIL] Output directory check failed")
    
    # Check 5: Idle monitoring
    try:
        import psutil
        readiness_score += 1
        print("[OK] System monitoring available")
    except ImportError:
        print("[FAIL] System monitoring not available")
    
    # Check 6: CLI functionality
    result = subprocess.run([
        "python", "emotion_quant_autopilot/quant_autopilot.py",
        "--config", "emotion_quant_autopilot/autopilot_config.json",
        "status"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        readiness_score += 1
        print("[OK] CLI functionality working")
    else:
        print("[FAIL] CLI functionality issues")
    
    # Final assessment
    print("\n" + "=" * 60)
    print(f"Integration Readiness: {readiness_score}/{total_checks} ({readiness_score/total_checks*100:.0f}%)")
    
    if readiness_score == total_checks:
        print("FULLY INTEGRATED - Ready for autonomous operation!")
        print("\nTo start autonomous quantization:")
        print("   python emotion_quant_autopilot/quant_autopilot.py --config emotion_quant_autopilot/autopilot_config.json start")
    elif readiness_score >= total_checks * 0.8:
        print("MOSTLY INTEGRATED - Ready with minor limitations")
    elif readiness_score >= total_checks * 0.6:
        print("PARTIALLY INTEGRATED - Some functionality may be limited")
    else:
        print("INTEGRATION INCOMPLETE - Significant issues need resolution")
    
    print("\nNext Steps:")
    print("   1. Idle system detection will trigger quantization")
    print("   2. Models will be quantized using integrated quantizer")
    print("   3. Emotional intelligence will be evaluated")
    print("   4. Results stored in enhanced database")
    print("   5. Best models marked as seed candidates")
    print("   6. System state persisted for crash recovery")
    
    return readiness_score == total_checks

if __name__ == "__main__":
    success = test_integrated_workflow()
    exit(0 if success else 1)
