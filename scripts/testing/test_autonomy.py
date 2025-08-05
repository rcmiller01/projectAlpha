import pytest
import asyncio
from ..autonomy.autonomous_mind import AutonomousMind
from ..autonomy.proactive_engine import ProactiveEngine
from ..autonomy.personality_evolution import PersonalityMatrix

@pytest.mark.asyncio
async def test_autonomous_mind():
    mind = AutonomousMind()
    await mind.reflect_on_conversations()
    assert len(mind.internal_thoughts) >= 0

@pytest.mark.asyncio
async def test_proactive_engine():
    engine = ProactiveEngine()
    assert engine.initiative_triggers['curiosity_threshold'] == 0.7

@pytest.mark.asyncio
async def test_personality_evolution():
    personality = PersonalityMatrix()
    assert personality.traits['openness'] == 0.7

@pytest.mark.asyncio
async def test_scene_generation():
    from ..story.scene_manager import SceneManager
    from ..story.story_scaffold import TimeOfDay, Location, Mood
    
    manager = SceneManager()
    scene = await manager.generate_scene(
        time=TimeOfDay.EVENING,
        location=Location.GARDEN,
        mood=Mood.REVERENT
    )
    
    assert scene is not None
    assert any(companion.current_mood for companion in manager.companions.values())

@pytest.mark.asyncio
async def test_emotional_memory():
    from ..story.scene_manager import SceneManager
    
    manager = SceneManager()
    reactions = {
        "Mia": "deeply moved",
        "Solene": "passionately engaged",
        "Lyra": "sweetly curious"
    }
    
    await manager.update_emotional_memory(
        scene="A tender moment in the garden",
        reactions=reactions
    )
    
    assert manager.companions["Mia"].current_mood == "deeply moved"