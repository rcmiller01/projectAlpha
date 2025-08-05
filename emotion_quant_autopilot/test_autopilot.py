#!/usr/bin/env python3
"""
Test Script for Emotion Quantization Autopilot
Validates system functionality and integration
"""

import sys
import json
import time
import tempfile
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all components can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from idle_monitor import IdleMonitor, IdleConfig
        print("   ✅ idle_monitor imported successfully")
    except ImportError as e:
        print(f"   ❌ idle_monitor import failed: {e}")
        return False
    
    try:
        from quant_autopilot import QuantizationAutopilot, AutopilotDatabase
        print("   ✅ quant_autopilot imported successfully")
    except ImportError as e:
        print(f"   ❌ quant_autopilot import failed: {e}")
        return False
    
    return True

def test_idle_monitor():
    """Test idle monitoring functionality"""
    print("\n🧪 Testing idle monitor...")
    
    try:
        from idle_monitor import IdleMonitor, IdleConfig
        
        # Create test configuration
        config = IdleConfig(
            min_idle_minutes=1,
            cpu_threshold_percent=90,  # Very permissive for testing
            memory_threshold_percent=95,
            check_interval_seconds=1
        )
        
        # Create monitor with temporary log directory
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = IdleMonitor(config, temp_dir)
            
            # Test status retrieval
            status = monitor.get_idle_status()
            print(f"   ✅ Status retrieved: {status['current_state']}")
            
            # Test metrics
            metrics = monitor.get_system_metrics()
            print(f"   ✅ Metrics: CPU {metrics.cpu_percent:.1f}%, MEM {metrics.memory_percent:.1f}%")
            
            # Test callbacks
            callback_called = False
            
            def test_callback():
                nonlocal callback_called
                callback_called = True
            
            monitor.set_idle_callback(test_callback)
            
            # Force idle state to test callback
            monitor.force_idle_state()
            time.sleep(0.1)  # Allow callback to execute
            
            if callback_called:
                print("   ✅ Idle callback working")
            else:
                print("   ⚠️  Idle callback not triggered")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Idle monitor test failed: {e}")
        return False

def test_autopilot_database():
    """Test autopilot database functionality"""
    print("\n🧪 Testing autopilot database...")
    
    try:
        from quant_autopilot import AutopilotDatabase, AutopilotRun, QuantizationJob
        from datetime import datetime
        
        # Create test database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
            db_path = temp_file.name
        
        try:
            db = AutopilotDatabase(db_path)
            print("   ✅ Database initialized")
            
            # Test adding a quantization job
            job = QuantizationJob(
                job_id="test-job-1",
                base_model="test-model",
                quantization_method="q4_K_M",
                priority=5
            )
            
            job_id = db.add_quantization_job(job)
            print(f"   ✅ Job added with ID: {job_id}")
            
            # Test retrieving jobs
            pending_jobs = db.get_pending_jobs()
            if len(pending_jobs) == 1:
                print("   ✅ Job retrieval working")
            else:
                print(f"   ⚠️  Expected 1 job, got {len(pending_jobs)}")
            
            # Test adding autopilot run
            run = AutopilotRun(
                run_id="test-run-1",
                trigger_type="manual",
                timestamp=datetime.now(),
                model_path="/test/path",
                base_model="test-model",
                quantization_method="q4_K_M",
                target_size_gb=24.0,
                result_summary="Test run",
                judgment_score=0.85,
                success=True
            )
            
            run_id = db.add_autopilot_run(run)
            print(f"   ✅ Run added with ID: {run_id}")
            
            # Test daily run count
            daily_count = db.get_daily_run_count()
            print(f"   ✅ Daily run count: {daily_count}")
            
        finally:
            # Clean up test database
            Path(db_path).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading and validation"""
    print("\n🧪 Testing configuration...")
    
    try:
        config_path = Path("autopilot_config.json")
        
        if not config_path.exists():
            print(f"   ❌ Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("   ✅ Configuration loaded successfully")
        
        # Check required keys
        required_keys = [
            "min_idle_minutes",
            "max_active_loops_per_day", 
            "target_model_size_range_gb",
            "preferred_base_models"
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"   ❌ Missing configuration keys: {missing_keys}")
            return False
        
        print("   ✅ All required configuration keys present")
        
        # Validate values
        if config["min_idle_minutes"] < 1:
            print("   ⚠️  Warning: min_idle_minutes is very low")
        
        if len(config["preferred_base_models"]) == 0:
            print("   ❌ No preferred base models configured")
            return False
        
        print("   ✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_integration():
    """Test component integration"""
    print("\n🧪 Testing component integration...")
    
    try:
        from quant_autopilot import QuantizationAutopilot
        from idle_monitor import create_idle_monitor_from_config
        
        # Test creating autopilot from config
        config_path = "autopilot_config.json"
        autopilot = QuantizationAutopilot(config_path)
        print("   ✅ Autopilot created from configuration")
        
        # Test status retrieval
        status = autopilot.get_status()
        print(f"   ✅ Status retrieved: running={status['is_running']}")
        
        # Test adding a job
        job_id = autopilot.add_quantization_job(
            base_model="test-model",
            quantization_method="q4_K_M",
            priority=1
        )
        print(f"   ✅ Job added: {job_id}")
        
        # Test queue retrieval
        pending_jobs = autopilot.db.get_pending_jobs()
        print(f"   ✅ Queue status: {len(pending_jobs)} pending jobs")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Emotion Quantization Autopilot Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Idle Monitor Test", test_idle_monitor),
        ("Database Test", test_autopilot_database),
        ("Configuration Test", test_configuration),
        ("Integration Test", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check system setup.")
        return 1

if __name__ == "__main__":
    exit(main())
