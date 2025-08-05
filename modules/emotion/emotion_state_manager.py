from .emotion_state import emotion_state

class EmotionStateManager:
    """Simple wrapper around global emotion_state providing accessor methods."""

    def get_current_mood(self) -> str:
        """Return the current dominant mood."""
        return emotion_state.current_emotion()


emotion_state_manager = EmotionStateManager()
