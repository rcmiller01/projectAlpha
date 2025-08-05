#!/usr/bin/env python3
"""Debug the CLI issue"""

import sys
import traceback
import json
from pathlib import Path

# Add the autopilot module to path
sys.path.insert(0, str(Path("emotion_quant_autopilot").absolute()))

try:
    print("Step 1: Importing...")
    from quant_autopilot import QuantizationAutopilot
    
    print("Step 2: Creating autopilot instance...")
    autopilot = QuantizationAutopilot("emotion_quant_autopilot/autopilot_config.json")
    
    print("Step 3: Checking logger attribute...")
    print(f"Has logger: {hasattr(autopilot, 'logger')}")
    if hasattr(autopilot, 'logger'):
        print(f"Logger type: {type(autopilot.logger)}")
    
    print("Step 4: Getting status...")
    status = autopilot.get_status()
    
    print("Step 5: Converting to JSON...")
    result = json.dumps(status, indent=2, default=str)
    
    print("Step 6: Success!")
    print(result)
    
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()
