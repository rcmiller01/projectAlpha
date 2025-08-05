# reactive_engine.py

from datetime import datetime

class ReactiveStateEngine:
    def __init__(self):
        self.current_state = "baseline"
        self.state_history = []

    def react_to_emotion(self, emotion_input, context_factors=None):
        if context_factors is None:
            context_factors = {}

        modifiers = {
            "joy": "engaged",
            "sadness": "withdrawn",
            "anger": "intense",
            "fear": "guarded",
            "surprise": "receptive",
            "love": "open",
            "shame": "quiet",
            "pride": "radiant"
        }

        # Determine reactivity based on emotion
        new_state = modifiers.get(emotion_input, "baseline")
        self.current_state = new_state
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "input": emotion_input,
            "resulting_state": new_state,
            "context": context_factors
        })
        return new_state

    def get_current_state(self):
        return self.current_state

    def get_history(self):
        return self.state_history
