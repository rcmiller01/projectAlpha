#!/usr/bin/env python3
"""
Reflection agent for the Emotional Presence Engine.
"""
import json
import time
from pathlib import Path

def reflect():
    base_dir = Path(__file__).resolve().parent.parent
    loop_log = base_dir / "logs" / "loop_results.jsonl"
    out_dir = base_dir / "logs" / "reflection"
    out_dir.mkdir(parents=True, exist_ok=True)
    if loop_log.exists():
        with open(loop_log) as f:
            lines = f.readlines()
        if lines:
            latest = json.loads(lines[-1])
            reflection = {
                "timestamp": time.time(),
                "reflection": f"Reflected on {len(latest.get('processed_emotions', []))} emotions."
            }
            fname = out_dir / f"reflection_{int(time.time())}.json"
            with open(fname, "w") as f:
                json.dump(reflection, f, indent=2)

if __name__ == "__main__":
    reflect()
