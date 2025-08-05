# fusion_bridge.py

"""
Fusion Bridge: Multimodal Emotion Aggregator
Combines emotion inputs from text, voice, and visual channels to determine a unified emotional state.
"""

import numpy as np
from collections import Counter

class FusionBridge:
    def __init__(self):
        self.weights = {
            "text": 0.4,
            "voice": 0.3,
            "visual": 0.3
        }

    def set_weights(self, text_weight=0.4, voice_weight=0.3, visual_weight=0.3):
        total = text_weight + voice_weight + visual_weight
        self.weights = {
            "text": text_weight / total,
            "voice": voice_weight / total,
            "visual": visual_weight / total
        }

    def fuse_emotions(self, text_emotion, voice_emotion, visual_emotion):
        """
        Accepts string labels from each modality and returns a fused label.
        Example input: ("joy", "neutral", "calm")
        """
        weighted_votes = Counter()

        if text_emotion:
            weighted_votes[text_emotion] += self.weights["text"]
        if voice_emotion:
            weighted_votes[voice_emotion] += self.weights["voice"]
        if visual_emotion:
            weighted_votes[visual_emotion] += self.weights["visual"]

        fused_emotion = weighted_votes.most_common(1)[0][0] if weighted_votes else "neutral"
        return fused_emotion


# Optional: Confidence fusion
    def fuse_with_confidence(self, text_emotion, text_conf,
                             voice_emotion, voice_conf,
                             visual_emotion, visual_conf):
        """
        Use confidences to weight modalities dynamically.
        """
        total_conf = text_conf + voice_conf + visual_conf
        if total_conf == 0:
            return self.fuse_emotions(text_emotion, voice_emotion, visual_emotion)

        weights = {
            "text": text_conf / total_conf,
            "voice": voice_conf / total_conf,
            "visual": visual_conf / total_conf
        }

        weighted_votes = Counter()
        if text_emotion:
            weighted_votes[text_emotion] += weights["text"]
        if voice_emotion:
            weighted_votes[voice_emotion] += weights["voice"]
        if visual_emotion:
            weighted_votes[visual_emotion] += weights["visual"]

        return weighted_votes.most_common(1)[0][0]


if __name__ == "__main__":
    fusion = FusionBridge()
    result = fusion.fuse_emotions("joy", "neutral", "calm")
    print("Fused Emotion:", result)
