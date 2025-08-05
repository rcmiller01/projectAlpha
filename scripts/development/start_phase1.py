#!/usr/bin/env python3
# start_phase1.py
# Startup script for Phase 1 Romantic AI System

import uvicorn
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required files exist"""
    required_files = [
        "modules/emotion/emotion_state.py",
        "modules/memory/mia_self_talk.py", 
        "modules/memory/mia_memory_response.py",
        "backend/main.py",
        "backend/romantic_routes.py",
        "config/mia_romantic.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def main():
    """Start the Phase 1 Romantic AI System"""
    print("💕 Starting Phase 1 Romantic AI Companion System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please ensure all required files are present before starting")
        sys.exit(1)
    
    print("\n🚀 Launching FastAPI server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation at: http://localhost:8000/docs")
    print("\n💕 Romantic API Endpoints:")
    print("   POST /api/romantic/interact - Romantic interactions")
    print("   GET  /api/romantic/status - Relationship status")
    print("   GET  /api/romantic/mia/thoughts - Mia's thoughts")
    print("   POST /api/romantic/milestone - Add milestones")
    print("   GET  /api/romantic/memories - Get memories")
    print("\n🧪 Test the system with: python test_phase1.py")
    print("=" * 60)
    
    try:
        # Start the server
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Phase 1 system...")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 