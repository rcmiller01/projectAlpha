from typing import Dict

from ..story.scene_manager import SceneManager
from .base import BaseLLM


class Storyteller(BaseLLM):
    async def generate_story_response(self, scene_context: dict, emotional_state: dict) -> str:
        manager = SceneManager()
        scene = await manager.generate_scene(
            time=scene_context.get("time"),
            location=scene_context.get("location"),
            mood=emotional_state.get("mood"),
        )

        await self.update_internal_state(scene)
        return scene
