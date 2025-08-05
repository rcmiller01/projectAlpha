#!/usr/bin/env python3
"""
Emotion Quantization Autopilot - Setup and Installation Script
Prepares the system for autonomous quantization execution
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict

def check_python_version() -> bool:
    """Check if Python version is 3.10+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.10+")
        return False

def check_dependencies() -> Dict[str, bool]:
    """Check if required Python packages are installed"""
    required_packages = [
        "psutil",
        "sqlite3",  # Built-in
        "pathlib",  # Built-in
        "logging",  # Built-in
        "threading",  # Built-in
        "subprocess",  # Built-in
        "json",  # Built-in
        "uuid",  # Built-in
        "datetime",  # Built-in
    ]
    
    optional_packages = [
        "pynput",  # For user activity monitoring
        "smtplib",  # For email notifications (built-in)
        "requests",  # For Slack notifications
    ]
    
    results = {}
    
    print("🔍 Checking required dependencies...")
    for package in required_packages:
        try:
            if package in ["sqlite3", "pathlib", "logging", "threading", "subprocess", "json", "uuid", "datetime"]:
                # These are built-in modules
                __import__(package)
                results[package] = True
                print(f"   ✅ {package}")
            else:
                __import__(package)
                results[package] = True
                print(f"   ✅ {package}")
        except ImportError:
            results[package] = False
            print(f"   ❌ {package} - MISSING")
    
    print("\n🔍 Checking optional dependencies...")
    for package in optional_packages:
        try:
            __import__(package)
            results[package] = True
            print(f"   ✅ {package}")
        except ImportError:
            results[package] = False
            print(f"   ⚠️  {package} - Optional (install for enhanced features)")
    
    return results

def install_missing_packages(missing_packages: List[str]) -> bool:
    """Install missing Python packages"""
    if not missing_packages:
        return True
    
    print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
    
    try:
        for package in missing_packages:
            if package in ["sqlite3", "pathlib", "logging", "threading", "subprocess", "json", "uuid", "datetime"]:
                continue  # Skip built-in modules
            
            print(f"   Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"   ✅ {package} installed")
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install packages: {e}")
        return False

def create_directory_structure() -> bool:
    """Create necessary directory structure"""
    directories = [
        "emotion_quant_autopilot/logs",
        "emotion_quant_autopilot/temp",
        "quant_pass1/models",
    ]
    
    print("\n📁 Creating directory structure...")
    
    try:
        for directory in directories:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Failed to create directories: {e}")
        return False

def validate_config_file() -> bool:
    """Validate autopilot configuration file"""
    config_path = Path("emotion_quant_autopilot/autopilot_config.json")
    
    print(f"\n⚙️  Validating configuration: {config_path}")
    
    if not config_path.exists():
        print(f"   ❌ Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required keys
        required_keys = [
            "min_idle_minutes",
            "max_active_loops_per_day",
            "target_model_size_range_gb",
            "preferred_base_models",
            "quantization_methods",
            "output_paths",
            "safety_limits"
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"   ❌ Missing configuration keys: {missing_keys}")
            return False
        
        # Validate specific settings
        if config["min_idle_minutes"] < 1:
            print("   ⚠️  Warning: min_idle_minutes is very low (< 1 minute)")
        
        if config["max_active_loops_per_day"] > 10:
            print("   ⚠️  Warning: max_active_loops_per_day is high (> 10)")
        
        if len(config["preferred_base_models"]) == 0:
            print("   ❌ No preferred base models configured")
            return False
        
        print("   ✅ Configuration file is valid")
        return True
    
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error validating configuration: {e}")
        return False

def check_existing_components() -> Dict[str, bool]:
    """Check if existing quantization components are available"""
    components = {
        "emotional_dataset_builder.py": Path("emotional_dataset_builder.py"),
        "emotion_training_tracker.py": Path("emotion_training_tracker.py"),
        "pass1_quantization_loop.py": Path("pass1_quantization_loop.py"),
        "idle_watchdog.py": Path("idle_watchdog.py"),
    }
    
    print("\n🔗 Checking existing components...")
    
    results = {}
    for name, path in components.items():
        if path.exists():
            results[name] = True
            print(f"   ✅ {name}")
        else:
            results[name] = False
            print(f"   ❌ {name} - NOT FOUND")
    
    return results

def create_emergency_stop_instructions() -> None:
    """Create emergency stop file and instructions"""
    emergency_file = Path("emotion_quant_autopilot/EMERGENCY_STOP")
    instructions_file = Path("emotion_quant_autopilot/EMERGENCY_STOP_INSTRUCTIONS.txt")
    
    print("\n🚨 Creating emergency stop mechanism...")
    
    # Create instructions file
    instructions = """
EMERGENCY STOP INSTRUCTIONS
===========================

To immediately stop all autopilot quantization activities:

1. Create the emergency stop file:
   touch emotion_quant_autopilot/EMERGENCY_STOP
   
   OR
   
   echo "STOP" > emotion_quant_autopilot/EMERGENCY_STOP

2. The autopilot will detect this file and stop scheduling new jobs.

3. Currently running quantization jobs will complete but no new jobs will start.

4. To resume operations, delete the emergency stop file:
   rm emotion_quant_autopilot/EMERGENCY_STOP

5. Monitor logs for confirmation:
   tail -f emotion_quant_autopilot/logs/autopilot_*.log

EMERGENCY CONTACTS:
- System Administrator: [Add contact info]
- Log Directory: emotion_quant_autopilot/logs/
- Configuration: emotion_quant_autopilot/autopilot_config.json

TROUBLESHOOTING:
- Check disk space: df -h
- Check running processes: ps aux | grep quantization
- Check system resources: htop or top
- View recent logs: tail -100 emotion_quant_autopilot/logs/autopilot_*.log
"""
    
    try:
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"   ✅ Emergency stop instructions created: {instructions_file}")
        print(f"   ℹ️  Emergency stop file path: {emergency_file}")
        
    except Exception as e:
        print(f"   ❌ Failed to create emergency stop instructions: {e}")

def create_systemd_service_template() -> None:
    """Create systemd service template for Linux systems"""
    if os.name != 'posix':
        return  # Skip on non-Unix systems
    
    service_template = """[Unit]
Description=Emotion Quantization Autopilot
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=YOUR_WORKING_DIRECTORY
ExecStart=YOUR_PYTHON_PATH YOUR_SCRIPT_PATH/emotion_quant_autopilot/quant_autopilot.py start
Restart=always
RestartSec=10

# Resource limits
MemoryMax=8G
CPUQuota=50%

# Security
NoNewPrivileges=true
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path("emotion_quant_autopilot/quantization-autopilot.service")
    
    try:
        with open(service_file, 'w') as f:
            f.write(service_template)
        
        print(f"\n🔧 Systemd service template created: {service_file}")
        print("   📝 Edit the template and copy to /etc/systemd/system/ to enable as a service")
        
    except Exception as e:
        print(f"   ❌ Failed to create systemd service template: {e}")

def run_basic_tests() -> bool:
    """Run basic functionality tests"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test idle monitor import
        sys.path.append("emotion_quant_autopilot")
        from idle_monitor import IdleMonitor, IdleConfig
        
        # Create test idle monitor
        config = IdleConfig(min_idle_minutes=1, check_interval_seconds=1)
        monitor = IdleMonitor(config, "emotion_quant_autopilot/logs")
        
        # Test status retrieval
        status = monitor.get_idle_status()
        
        print("   ✅ Idle monitor test passed")
        
        # Test autopilot import
        from quant_autopilot import QuantizationAutopilot, AutopilotDatabase
        
        # Test database initialization
        db = AutopilotDatabase("emotion_quant_autopilot/test_autopilot.db")
        
        print("   ✅ Autopilot database test passed")
        
        # Clean up test database
        test_db_path = Path("emotion_quant_autopilot/test_autopilot.db")
        if test_db_path.exists():
            test_db_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic tests failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Emotion Quantization Autopilot Setup")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version requirement not met")
        return 1
    
    # Step 2: Check dependencies
    dependencies = check_dependencies()
    missing_required = [pkg for pkg, available in dependencies.items() 
                       if not available and pkg in ["psutil"]]
    
    if missing_required:
        print(f"\n📦 Installing missing required packages...")
        if not install_missing_packages(missing_required):
            print("\n❌ Setup failed: Could not install required packages")
            return 1
    
    # Step 3: Create directory structure
    if not create_directory_structure():
        print("\n❌ Setup failed: Could not create directories")
        return 1
    
    # Step 4: Validate configuration
    if not validate_config_file():
        print("\n❌ Setup failed: Configuration validation failed")
        return 1
    
    # Step 5: Check existing components
    components = check_existing_components()
    missing_components = [name for name, available in components.items() if not available]
    
    if missing_components:
        print(f"\n⚠️  Warning: Missing components: {missing_components}")
        print("   Some features may not work correctly")
    
    # Step 6: Create emergency stop mechanism
    create_emergency_stop_instructions()
    
    # Step 7: Create systemd service template (Linux only)
    create_systemd_service_template()
    
    # Step 8: Run basic tests
    if not run_basic_tests():
        print("\n⚠️  Warning: Basic tests failed - check system setup")
    
    # Final summary
    print("\n" + "=" * 50)
    print("✅ Setup completed!")
    print("\n🔧 Next steps:")
    print("1. Review configuration: emotion_quant_autopilot/autopilot_config.json")
    print("2. Test the system: python emotion_quant_autopilot/quant_autopilot.py status")
    print("3. Start autopilot: python emotion_quant_autopilot/quant_autopilot.py start")
    print("4. Monitor logs: tail -f emotion_quant_autopilot/logs/autopilot_*.log")
    print("\n🚨 Emergency stop: touch emotion_quant_autopilot/EMERGENCY_STOP")
    
    return 0

if __name__ == "__main__":
    exit(main())
