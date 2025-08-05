
import random

def score_importance(emotion, symbol=None):
    base = {
        "longing": 0.9,
        "devoted": 0.95,
        "joy": 0.7,
        "vulnerable": 0.85,
        "reflective": 0.75,
        "anxious": 0.6,
        "neutral": 0.5
    }.get(emotion, 0.6)

    if symbol in ["collar", "garden", "ring"]:
        base += 0.1
    return round(min(1.0, base + random.uniform(-0.05, 0.05)), 4)

def link_to_thoughts(thoughts_list, memory_text):
    return [t for t in thoughts_list if any(kw in memory_text.lower() for kw in t.lower().split())]
