#!/usr/bin/env python3
import json, subprocess
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
with open(base_dir / "config" / "council_manifest.json") as f:
    loops = json.load(f)["loops"]
for loop in loops:
    script = base_dir / "agents" / loop
    if script.exists():
        subprocess.run(["python3", str(script)])
    else:
        print(f"Missing council loop: {loop}")
