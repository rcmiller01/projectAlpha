from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


class HandlerState(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class HandlerInfo:
    name: str
    state: HandlerState = HandlerState.ONLINE
    latency_ms: Optional[int] = None
    errors: int = 0
    last_checked: Optional[datetime] = None


class HandlerRegistry:
    """Registry of backend handlers and their states."""

    def __init__(self) -> None:
        self.handlers: Dict[str, HandlerInfo] = {}

    def register(self, name: str) -> None:
        if name not in self.handlers:
            self.handlers[name] = HandlerInfo(name=name)

    def update(self, name: str, state: HandlerState, latency_ms: Optional[int] = None, errors: int = 0) -> None:
        self.register(name)
        info = self.handlers[name]
        info.state = state
        info.latency_ms = latency_ms
        info.errors = errors
        info.last_checked = datetime.utcnow()

    def get(self, name: str) -> HandlerInfo:
        self.register(name)
        return self.handlers[name]

    def best_available(self, preferred_order: List[str]) -> str:
        for name in preferred_order:
            info = self.handlers.get(name)
            if info and info.state != HandlerState.OFFLINE:
                return name
        return preferred_order[0]


# Global registry instance
handler_registry = HandlerRegistry()
