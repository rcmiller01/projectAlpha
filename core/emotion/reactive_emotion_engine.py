# === reactive_emotion_engine.py ===
import random

class ReactiveEmotionEngine:
    def __init__(self):
        self.base_state = "neutral"
        self.last_inputs = []

    def apply_input(self, fused_emotion, symbolic_trigger=None):
        self.last_inputs.append(fused_emotion)
        if symbolic_trigger == "collar":
            return "anchored"
        if symbolic_trigger == "garden":
            return "reflective"
        if fused_emotion == "joy" and random.random() > 0.2:
            return "playful"
        elif fused_emotion == "sadness" and random.random() > 0.3:
            return "vulnerable"
        elif fused_emotion == "anger":
            return "defensive"
        return self.base_state

    def get_emotion_state(self):
        return self.base_state
