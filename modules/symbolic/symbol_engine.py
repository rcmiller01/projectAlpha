import json
import os
from datetime import datetime

ACTIVE_SYMBOL = {
    "symbol": None,
    "timestamp": None
}

def load_symbol_definitions():
    path = os.path.join("config", "symbol_triggers.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Symbol Engine] Failed to load: {e}")
        return {}

def trigger_symbol(symbol_name):
    symbols = load_symbol_definitions()
    if symbol_name not in symbols:
        print(f"[Symbol Engine] Unknown symbol: {symbol_name}")
        return False

    ACTIVE_SYMBOL["symbol"] = symbol_name
    ACTIVE_SYMBOL["timestamp"] = datetime.now().isoformat()

    print(f"[Symbol Triggered] → {symbol_name}")
    return True

def is_symbol_active():
    return ACTIVE_SYMBOL["symbol"] is not None

def clear_symbol():
    ACTIVE_SYMBOL["symbol"] = None
    ACTIVE_SYMBOL["timestamp"] = None
