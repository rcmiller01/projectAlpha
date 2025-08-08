import sys

sys.path.append(".")
from datetime import datetime

from modules.memory_writer import log_memory_entry
from modules.symbolic_ritual_manager import execute


def startup_sequence():
    try:
        execute("rebirth")
        log_memory_entry({"event": "rebirth", "timestamp": datetime.now().isoformat()})
    except Exception as e:
        print(f"Startup sequence failed: {e}")
        log_memory_entry(
            {"event": "rebirth_failure", "timestamp": datetime.now().isoformat(), "error": str(e)}
        )


if __name__ == "__main__":
    startup_sequence()
