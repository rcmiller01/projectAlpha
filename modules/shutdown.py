from symbolic_ritual_manager import execute
from memory_writer import log_memory_entry
from datetime import datetime

def shutdown_sequence():
    try:
        execute("goodbye")
        log_memory_entry({
            "event": "goodbye",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Shutdown sequence failed: {e}")
        log_memory_entry({
            "event": "goodbye_failure",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })

if __name__ == "__main__":
    shutdown_sequence()
