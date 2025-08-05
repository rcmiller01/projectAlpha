"""
Local JSON Database Implementation

Provides simple persistent storage using a JSON file.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .database_interface import (
    DatabaseInterface,
    UserProfile,
    InteractionRecord,
    SessionRecord,
    PsychologicalState,
    MemoryFragment,
)


class JSONDatabase:
    """Simple JSON file based database for persistence."""

    def __init__(self, path: str):
        self.path = path
        self.logger = logging.getLogger(__name__)
        self._data: Dict[str, Any] = {
            "users": {},
            "interactions": [],
            "sessions": {},
            "psychological_states": [],
            "memory_fragments": [],
        }

    async def initialize(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            self.logger.info("JSON database loaded")
        except FileNotFoundError:
            self.logger.info("JSON database file not found, starting new one")
        except Exception as e:
            self.logger.error(f"Error loading JSON database: {e}")

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving JSON database: {e}")

    async def create_user_profile(self, user_profile: UserProfile) -> bool:
        if user_profile.user_id in self._data["users"]:
            return False
        self._data["users"][user_profile.user_id] = user_profile.to_dict()
        self._save()
        return True

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        data = self._data["users"].get(user_id)
        return UserProfile.from_dict(data) if data else None

    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        if user_id not in self._data["users"]:
            return False
        self._data["users"][user_id].update(updates)
        self._data["users"][user_id]["last_active"] = datetime.now().isoformat()
        self._save()
        return True

    async def save_interaction(self, interaction: InteractionRecord) -> bool:
        self._data["interactions"].append(interaction.to_dict())
        self._save()
        return True

    async def get_recent_interactions(self, user_id: str, limit: int = 20) -> List[InteractionRecord]:
        records = [r for r in self._data["interactions"] if r["user_id"] == user_id]
        records.sort(key=lambda x: x["timestamp"], reverse=True)
        return [InteractionRecord.from_dict(r) for r in records[:limit]]

    async def save_psychological_state(self, state: PsychologicalState) -> bool:
        self._data["psychological_states"].append(state.to_dict())
        self._save()
        return True

    async def get_latest_psychological_state(self, user_id: str) -> Optional[PsychologicalState]:
        states = [s for s in self._data["psychological_states"] if s["user_id"] == user_id]
        if not states:
            return None
        states.sort(key=lambda x: x["timestamp"], reverse=True)
        return PsychologicalState.from_dict(states[0])

    async def save_memory_fragment(self, memory: MemoryFragment) -> bool:
        self._data["memory_fragments"].append(memory.to_dict())
        self._save()
        return True

    async def get_relevant_memories(self, user_id: str, memory_type: Optional[str] = None,
                                    tags: Optional[List[str]] = None, limit: int = 10) -> List[MemoryFragment]:
        memories = [m for m in self._data["memory_fragments"] if m["user_id"] == user_id]
        if memory_type:
            memories = [m for m in memories if m["memory_type"] == memory_type]
        if tags:
            memories = [m for m in memories if any(tag in m.get("tags", []) for tag in tags)]
        memories.sort(key=lambda x: x.get("importance_score", 0), reverse=True)
        return [MemoryFragment.from_dict(m) for m in memories[:limit]]

    async def close(self):
        self._save()
        self.logger.info("JSON database saved")
