"""Inflection-to-Voice bridge linking mood and scene intent to voice style."""

from typing import Dict

from ...backend.modules.voice.voice_orchestrator import voice_orchestrator, VoiceEngine
from ..emotion.emotion_state_manager import emotion_state_manager
from ..sensory_preference import sensory_preferences


STYLE_MAP: Dict[str, Dict[str, str]] = {
    "romantic": {
        "tender": "breathy",
        "passionate": "sultry",
        "reverent": "reverent",
    },
    "casual": {
        "playful": "cheerful",
        "neutral": "plain",
    },
}


class VoiceManager:
    def __init__(self):
        self.default_engine = VoiceEngine.PIPER_TTS

    def mood_to_style(self, mood: str, intent: str) -> str:
        return STYLE_MAP.get(intent, {}).get(mood, "plain")

    async def speak(self, text: str, scene_intent: str) -> bytes:
        mood = emotion_state_manager.get_current_mood()
        style = self.mood_to_style(mood, scene_intent)
        modifiers = sensory_preferences.influence_voice_style("default")
        try:
            profile = voice_orchestrator.get_voice_profile()
            profile.update(modifiers)
            voice_orchestrator.update_voice_profile(
                pitch=profile.get("pitch"), speed=profile.get("speed")
            )
            audio = await voice_orchestrator.tts_engines[self.default_engine].synthesize(text, style)
            if audio:
                return audio
        except Exception:
            pass
        # Fallback simple TTS
        return text.encode()


voice_manager = VoiceManager()
