# cbt_scaffold.py

"""
CBT Scaffold for AI Emotional Companion
- Inspired by Woebot-style interventions
- Offers simple cognitive reframing, validation, and emotional grounding
- Meant to be called from mood engine or dialogue system when distress is detected
"""

import random
from datetime import datetime

# Define basic CBT tools
def cognitive_reframe(negative_thought):
    reframes = [
        "What might you say to a friend thinking this?",
        "Is there another way to interpret that event?",
        "What evidence supports this thought? What evidence contradicts it?",
        "Could this be a feeling talking instead of a fact?"
    ]
    chosen = random.choice(reframes)
    return f"Let's pause and reflect. {chosen}"

def emotional_validation(emotion):
    templates = [
        f"It's okay to feel {emotion}. That feeling is valid.",
        f"Many people feel {emotion} in moments like this. You're not alone.",
        f"{emotion.title()} might be heavy right now, but it won't last forever."
    ]
    return random.choice(templates)

def grounding_exercise():
    return "Let's try a quick grounding exercise: Name 5 things you see, 4 things you can touch, 3 things you hear, 2 things you smell, and 1 thing you taste."

# Intervention trigger entry point
def offer_intervention(mood_state, recent_thought=None):
    timestamp = datetime.now().isoformat()
    if mood_state in ["anxious", "overwhelmed", "low", "numb"]:
        response = emotional_validation(mood_state)
        if recent_thought:
            reframe = cognitive_reframe(recent_thought)
            return {
                "timestamp": timestamp,
                "intervention": "cbt_combo",
                "response": f"{response} {reframe}"
            }
        return {
            "timestamp": timestamp,
            "intervention": "cbt_validate",
            "response": response
        }
    elif mood_state in ["dissociated", "flat"]:
        return {
            "timestamp": timestamp,
            "intervention": "cbt_grounding",
            "response": grounding_exercise()
        }
    else:
        return {
            "timestamp": timestamp,
            "intervention": "none",
            "response": "No intervention needed right now."
        }

# Example usage
if __name__ == "__main__":
    test = offer_intervention("anxious", "I think I always mess things up")
    print(test)
