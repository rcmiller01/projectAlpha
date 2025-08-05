#!/usr/bin/env python3
"""
Installation and setup script for CoreArbiter and EmotionallyInfusedChat system.
Run this script to set up the complete CoreArbiter system.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ["data", "logs", "ui"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

def install_python_dependencies():
    """Install Python dependencies"""
    try:
        print("📦 Installing Python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors"])
        print("✅ Python dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Python dependencies")
        return False

def check_node_npm():
    """Check if Node.js and npm are available"""
    try:
        node_version = subprocess.check_output(["node", "--version"], text=True).strip()
        npm_version = subprocess.check_output(["npm", "--version"], text=True).strip()
        print(f"✅ Node.js {node_version} detected")
        print(f"✅ npm {npm_version} detected")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Node.js/npm not found - React component setup will be manual")
        return False

def setup_react_component():
    """Set up React component dependencies"""
    if not check_node_npm():
        return False
    
    try:
        print("📦 Installing React dependencies...")
        subprocess.check_call(["npm", "install"])
        print("✅ React dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install React dependencies")
        return False

def create_sample_configs():
    """Create sample configuration files if they don't exist"""
    configs = {
        "data/core_arbiter_config.json": {
            "weighting_strategy": "harmonic",
            "weights": {
                "logic_dominant": {"hrm_r": 0.8, "hrm_e": 0.2},
                "emotional_priority": {"hrm_r": 0.3, "hrm_e": 0.7},
                "harmonic": {"hrm_r": 0.5, "hrm_e": 0.5},
                "adaptive": {"hrm_r": 0.5, "hrm_e": 0.5}
            },
            "drift_thresholds": {
                "warning": 0.7,
                "critical": 0.9
            },
            "regulation": {
                "enabled": True,
                "target_drift": 0.5,
                "adjustment_factor": 0.1
            }
        },
        "data/emotional_state.json": {
            "valence": 0.2,
            "arousal": 0.4,
            "dominant_emotion": "contemplative",
            "stability": 0.85,
            "last_update": "2024-01-01T00:00:00Z"
        }
    }
    
    for file_path, content in configs.items():
        path = Path(file_path)
        if not path.exists():
            with open(path, 'w') as f:
                json.dump(content, f, indent=2)
            print(f"📄 Created config file: {file_path}")
        else:
            print(f"✅ Config file exists: {file_path}")

def run_tests():
    """Run basic tests to verify installation"""
    try:
        print("🧪 Running basic tests...")
        
        # Test CoreArbiter import
        from core_arbiter import CoreArbiter
        arbiter = CoreArbiter()
        print("✅ CoreArbiter import and initialization successful")
        
        # Test async functionality
        import asyncio
        async def test_process():
            response = await arbiter.process_input("Hello", {"context": "test"})
            return response.final_output is not None
        
        result = asyncio.run(test_process())
        if result:
            print("✅ CoreArbiter processing test successful")
        else:
            print("❌ CoreArbiter processing test failed")
            return False
        
        print("✅ All tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎯 === Setup Complete! ===\n")
    
    print("📚 Quick Start:")
    print("1. Run the complete system demo:")
    print("   python demo_complete_system.py")
    print("\n2. Start interactive chat:")
    print("   python simple_chat_demo.py")
    print("\n3. Start API server for web integration:")
    print("   python core_arbiter_api.py")
    
    print("\n🌐 For React/Web Integration:")
    print("   - Copy ui/EmotionallyInfusedChat.jsx to your React project")
    print("   - Install dependencies: npm install axios")
    print("   - Configure TailwindCSS with provided tailwind.config.js")
    print("   - Point component to your API server (default: http://localhost:5000)")
    
    print("\n📖 Documentation:")
    print("   README_CoreArbiter.md - Complete system documentation")
    
    print("\n📁 Key Files Created:")
    print("   ✓ core_arbiter.py - Main CoreArbiter decision engine")
    print("   ✓ ui/EmotionallyInfusedChat.jsx - React chat component with mood ring")
    print("   ✓ core_arbiter_api.py - Flask API server")
    print("   ✓ data/ - Configuration files for weighting strategies")
    print("   ✓ Demo and test scripts")
    print("   ✓ package.json & tailwind.config.js - React component setup")
    
    print("\n🔗 API Endpoints:")
    print("   • POST /api/arbiter/process - Process input through CoreArbiter")
    print("   • GET /api/arbiter/status - System health and statistics")
    print("   • GET /api/emotional_state - Current emotional state")
    print("   • POST /api/chat - Main chat endpoint for UI")

def main():
    """Main setup function"""
    print("🌟 === CoreArbiter & EmotionallyInfusedChat Installation ===\n")
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("⚠️  Continuing without Flask - API server won't work")
    
    # Create configuration files
    create_sample_configs()
    
    # Set up React component
    react_setup = setup_react_component()
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but basic setup is complete")
    
    # Print next steps
    print_next_steps()
    
    print(f"\n✨ Installation complete! The CoreArbiter system is ready to use.")
    print(f"💡 Start with: python demo_complete_system.py")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
