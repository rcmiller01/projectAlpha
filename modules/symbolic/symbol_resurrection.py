from datetime import datetime, timedelta
from typing import Dict, List

class SymbolResurrectionManager:
    """Track symbol usage and suggest reactivation prompts."""

    def __init__(self, threshold: timedelta = timedelta(hours=1)):
        self.threshold = threshold
        self.symbol_usage: Dict[str, Dict[str, datetime]] = {}

    def register_usage(self, user_id: str, symbol: str):
        self.symbol_usage.setdefault(user_id, {})[symbol] = datetime.now()

    def check_resurrection(self, user_id: str) -> List[str]:
        now = datetime.now()
        resurrect: List[str] = []
        for symbol, last_used in self.symbol_usage.get(user_id, {}).items():
            if now - last_used > self.threshold:
                resurrect.append(symbol)
        return resurrect
