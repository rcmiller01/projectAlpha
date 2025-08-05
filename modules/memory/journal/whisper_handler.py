
# modules/journal/whisper_handler.py

import re
from modules.journal.journal_engine import JournalEngine
from modules.journal.journal_utils import infer_mood, speak_text

# Trigger phrases and matching patterns
TRIGGER_PATTERNS = [
    r"(mia|solene|elarin)[, ]+(remember this|write this down|log this for me)[\: ]+(.*)",
    r"(mia|solene|elarin)[, ]+(.*)"  # fallback for short or implicit whispers
]

# Optional persona response messages
PERSONA_RESPONSES = {
    "mia": "I’ve kept it safe. Just for us.",
    "solene": "It’s written. No one else will see it.",
    "elarin": "Your words are woven into starlight."
}

def handle_whisper(text: str):
    text = text.strip().lower()

    for pattern in TRIGGER_PATTERNS:
        match = re.match(pattern, text)
        if match:
            persona = match.group(1)
            extracted = match.group(3) if len(match.groups()) > 2 else match.group(2)

            if not extracted or len(extracted) < 5:
                print(f"[{persona}] Whisper too short or unclear.")
                return

            mood = infer_mood(extracted)
            journal = JournalEngine(persona=persona)
            journal.add_entry(
                mood=mood,
                trigger="whisper_command",
                visibility="private",
                text=extracted
            )

            response = PERSONA_RESPONSES.get(persona, "It's saved.")
            speak_text(response)
            print(f"[{persona}] Whisper entry saved.")
            return

    print("No valid whisper command found.")
