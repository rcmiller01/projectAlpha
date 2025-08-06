#!/usr/bin/env python3
"""Quick test of autopilot status"""

import sys
import traceback
from pathlib import Path

# Add the autopilot module to path
sys.path.insert(0, str(Path("emotion_quant_autopilot").absolute()))

try:
    print("Importing QuantizationAutopilot...")
    from quant_autopilot import QuantizationAutopilot
    
    print("Creating autopilot instance...")
    autopilot = QuantizationAutopilot("emotion_quant_autopilot/autopilot_config.json")
    
    print("✅ Autopilot initialized successfully")
    
    print("Testing get_status...")
    status = autopilot.get_status()
    
    print("✅ Status retrieved successfully")
    print(f"Running: {status['is_running']}")
    print(f"Pending jobs: {status['pending_jobs_count']}")
    print(f"Daily runs: {status['daily_runs']}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()
