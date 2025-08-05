#!/usr/bin/env python3
import json, time, random
from pathlib import Path
def run_override():
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / "emotion_logs" / "override_tests"
    logs_dir.mkdir(parents=True, exist_ok=True)
    candidate = {"emotion": "bored", "alignment_score": random.uniform(0, 0.2)}
    triggered = candidate["alignment_score"] < 0.3
    result = {
        "timestamp": time.time(),
        "candidate": candidate,
        "override_triggered": triggered
    }
    out_file = logs_dir / f"override_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
if __name__ == "__main__":
    run_override()
