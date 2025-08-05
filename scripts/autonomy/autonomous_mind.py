from typing import Dict

from .base import BaseLLM
from ..story.scene_manager import SceneManager

class Storyteller(BaseLLM):
    async def generate_story_response(self, 
                                    scene_context: Dict,
                                    emotional_state: Dict) -> str:
        
        manager = SceneManager()
        scene = await manager.generate_scene(
            time=scene_context.get('time'),
            location=scene_context.get('location'),
            mood=emotional_state.get('mood')
        )
        
        await self.update_internal_state(scene)
        return scene