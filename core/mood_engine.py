from modules.utils.emotion_utils import analyze_text_emotion


def evaluate_mood(text: str):
    """Simple wrapper that analyzes text and returns a list of moods."""
    result = analyze_text_emotion(text)
    mood = result.get("emotion")
    return [mood] if mood else [] 