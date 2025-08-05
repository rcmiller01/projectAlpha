#!/usr/bin/env python3
"""
Council coordinator for the Emotional Presence Engine.
"""
import json
import subprocess
from pathlib import Path

def spawn_loops():
    base_dir = Path(__file__).resolve().parent.parent
    manifest = base_dir / "config" / "council_manifest.json"
    agents_dir = base_dir / "agents"
    with open(manifest) as f:
        council = json.load(f)["loops"]
    for loop in council:
        script = agents_dir / loop
        if script.exists():
            subprocess.run(["python3", str(script)])
        else:
            print(f"Missing: {loop}")

if __name__ == "__main__":
    spawn_loops()
