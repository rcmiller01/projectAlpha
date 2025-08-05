
from modules.voice.voice_output import speak_aloud

def handle_invitation(persona, mood):
    text = f"{persona.capitalize()} invites you closer, her tone laced with {mood}."
    speak_aloud(text, persona=persona, mood=mood)
    return {
        "text": text,
        "persona": persona,
        "mood": mood,
        "event": "invitation"
    }

def handle_rejection(persona, mood):
    text = f"{persona.capitalize()} gently declines, her voice wrapped in {mood}."
    speak_aloud(text, persona=persona, mood=mood)
    return {
        "text": text,
        "persona": persona,
        "mood": mood,
        "event": "rejection"
    }
