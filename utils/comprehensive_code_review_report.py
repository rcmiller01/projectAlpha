#!/usr/bin/env python3
"""
Comprehensive Code Review and Testing Report
Generates a complete analysis of the AI companion system
"""

import importlib.util
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


def test_module_imports():
    """Test importing key modules"""
    modules = {
        "core_arbiter.py": "CoreArbiter",
        "emotion_loop_core.py": "EmotionLoopManager",
        "dolphin_backend.py": "app",
        "autopilot_bootloader.py": "AutopilotBootloader",
        "api_bridge.py": "app",
    }

    results = {}
    for file_path, expected_class in modules.items():
        try:
            if os.path.exists(file_path):
                spec = importlib.util.spec_from_file_location(
                    file_path.replace(".py", ""), file_path
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                # Check if expected class/object exists
                if hasattr(mod, expected_class):
                    results[file_path] = {
                        "status": "SUCCESS",
                        "message": f"Module imported, {expected_class} found",
                    }
                else:
                    results[file_path] = {
                        "status": "WARNING",
                        "message": f"Module imported but {expected_class} not found",
                    }
            else:
                results[file_path] = {"status": "ERROR", "message": "File not found"}
        except Exception as e:
            results[file_path] = {"status": "ERROR", "message": str(e)[:100]}

    return results


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "requests",
        "pydantic",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "flask",
        "flask-cors",
    ]

    results = {}
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            results[package] = {"status": "INSTALLED", "message": "Available"}
        except ImportError:
            results[package] = {"status": "MISSING", "message": "Not installed"}

    return results


def analyze_file_structure():
    """Analyze project file structure"""
    important_files = [
        "README.md",
        "requirements.txt",
        "package.json",
        "core_arbiter.py",
        "emotion_loop_core.py",
        "dolphin_backend.py",
    ]

    important_dirs = ["modules/", "backend/", "data/", "config/", "scripts/"]

    file_results = {}
    for file in important_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            file_results[file] = {"status": "EXISTS", "size": f"{size} bytes", "message": "Present"}
        else:
            file_results[file] = {"status": "MISSING", "size": "0 bytes", "message": "Not found"}

    dir_results = {}
    for directory in important_dirs:
        if os.path.exists(directory):
            file_count = len(
                [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            )
            dir_results[directory] = {
                "status": "EXISTS",
                "files": file_count,
                "message": f"{file_count} files",
            }
        else:
            dir_results[directory] = {
                "status": "MISSING",
                "files": 0,
                "message": "Directory not found",
            }

    return file_results, dir_results


def check_configuration_files():
    """Check configuration files"""
    config_files = [
        "config/anchor_settings.json",
        "bootloader_config.json",
        "drift_config.json",
        "emotional_prompts.json",
    ]

    results = {}
    for config_file in config_files:
        try:
            if os.path.exists(config_file):
                with open(config_file) as f:
                    data = json.load(f)
                    results[config_file] = {
                        "status": "VALID",
                        "keys": len(data) if isinstance(data, dict) else "N/A",
                        "message": "Valid JSON",
                    }
            else:
                results[config_file] = {"status": "MISSING", "keys": 0, "message": "File not found"}
        except json.JSONDecodeError:
            results[config_file] = {
                "status": "INVALID",
                "keys": 0,
                "message": "Invalid JSON format",
            }
        except Exception as e:
            results[config_file] = {"status": "ERROR", "keys": 0, "message": str(e)[:50]}

    return results


def generate_report():
    """Generate comprehensive report"""
    print(
        f"""
{'='*80}
🔍 COMPREHENSIVE CODE REVIEW & TESTING REPORT
{'='*80}
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏠 Project: AI Companion System (projectAlpha)
📍 Directory: {os.getcwd()}
{'='*80}
"""
    )

    # Test module imports
    print("🔧 MODULE IMPORT ANALYSIS")
    print("-" * 40)
    import_results = test_module_imports()
    success_count = sum(1 for r in import_results.values() if r["status"] == "SUCCESS")
    total_count = len(import_results)

    for module, result in import_results.items():
        status_icon = (
            "✅"
            if result["status"] == "SUCCESS"
            else "⚠️"
            if result["status"] == "WARNING"
            else "❌"
        )
        print(f"{status_icon} {module:<25} | {result['message']}")

    print(
        f"\n📊 Import Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)"
    )

    # Check dependencies
    print("\n🔗 DEPENDENCY ANALYSIS")
    print("-" * 40)
    dep_results = check_dependencies()
    installed_count = sum(1 for r in dep_results.values() if r["status"] == "INSTALLED")
    total_deps = len(dep_results)

    for package, result in dep_results.items():
        status_icon = "✅" if result["status"] == "INSTALLED" else "❌"
        print(f"{status_icon} {package:<20} | {result['message']}")

    print(
        f"\n📊 Dependencies: {installed_count}/{total_deps} ({installed_count/total_deps*100:.1f}%)"
    )

    # File structure analysis
    print("\n📁 FILE STRUCTURE ANALYSIS")
    print("-" * 40)
    file_results, dir_results = analyze_file_structure()

    print("📄 Important Files:")
    file_exist_count = sum(1 for r in file_results.values() if r["status"] == "EXISTS")
    for file, result in file_results.items():
        status_icon = "✅" if result["status"] == "EXISTS" else "❌"
        print(f"  {status_icon} {file:<25} | {result['size']:<15} | {result['message']}")

    print("\n📂 Important Directories:")
    dir_exist_count = sum(1 for r in dir_results.values() if r["status"] == "EXISTS")
    for directory, result in dir_results.items():
        status_icon = "✅" if result["status"] == "EXISTS" else "❌"
        print(f"  {status_icon} {directory:<25} | {result['files']:<5} files | {result['message']}")

    print(
        f"\n📊 File Structure: Files {file_exist_count}/{len(file_results)}, Dirs {dir_exist_count}/{len(dir_results)}"
    )

    # Configuration analysis
    print("\n⚙️ CONFIGURATION ANALYSIS")
    print("-" * 40)
    config_results = check_configuration_files()
    valid_configs = sum(1 for r in config_results.values() if r["status"] == "VALID")

    for config, result in config_results.items():
        status_icon = (
            "✅" if result["status"] == "VALID" else "⚠️" if result["status"] == "MISSING" else "❌"
        )
        keys_info = f"({result['keys']} keys)" if result["keys"] else ""
        print(f"  {status_icon} {config:<30} | {result['message']} {keys_info}")

    print(
        f"\n📊 Configuration: {valid_configs}/{len(config_results)} ({valid_configs/len(config_results)*100:.1f}%)"
    )

    # Overall assessment
    print(f"\n{'='*80}")
    print("🎯 OVERALL SYSTEM ASSESSMENT")
    print("=" * 80)

    overall_health = (
        (success_count / total_count * 0.3)
        + (installed_count / total_deps * 0.3)
        + (file_exist_count / len(file_results) * 0.2)
        + (dir_exist_count / len(dir_results) * 0.1)
        + (valid_configs / len(config_results) * 0.1)
    ) * 100

    health_status = (
        "🟢 EXCELLENT"
        if overall_health >= 90
        else "🟡 GOOD"
        if overall_health >= 75
        else "🟠 FAIR"
        if overall_health >= 60
        else "🔴 NEEDS WORK"
    )

    print(f"Overall Health Score: {overall_health:.1f}% - {health_status}")

    print("\n🎉 RECOMMENDATIONS:")
    if success_count < total_count:
        print("   • Fix module import issues for better system stability")
    if installed_count < total_deps:
        print("   • Install missing dependencies to enable full functionality")
    if valid_configs < len(config_results):
        print("   • Verify and fix configuration files")
    if overall_health >= 75:
        print("   • System is in good condition and ready for use")
        print("   • Consider running integration tests for final verification")

    print(f"\n{'='*80}")
    print("Report generated successfully! 🎊")
    print("=" * 80)


if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        traceback.print_exc()
