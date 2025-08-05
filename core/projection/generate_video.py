
# modules/projection/generate_video.py

import os
from datetime import datetime
from modules.projection.symbolic_overlays import get_symbolic_effects
from modules.projection.consent_manager import is_render_allowed, is_selective
from modules.projection.rejection_phrases import get_rejection_phrase
from modules.projection.invitation_phrases import get_invitation_phrase
from modules.journal.journal_engine import JournalEngine

def generate_video_prompt(persona, mood, state, journal_text, allow_nsfw=False):
    symbols = get_symbolic_effects(state)
    overlay_phrases = [f"{k.replace('_', ' ')} glowing with {v}" for k, v in symbols.items() if k != "none"]
    emotional_description = f"{persona} in a state of {mood}, responding to journal entry: '{journal_text.strip()}'"
    nsfw_tag = "Explicit detail, soft motion, erotic nuance, sensual breathing, intimate rhythm" if allow_nsfw else ""
    prompt = f"{emotional_description}, " + ", ".join(overlay_phrases)
    if allow_nsfw:
        prompt += ", " + nsfw_tag
    return prompt.strip()

def generate_video(persona="mia", mood="devoted", state="devoted", journal_text="", allow_nsfw=False, user="robert"):
    journal = JournalEngine(persona)

    if not is_render_allowed(persona, mood, allow_nsfw, user):
        phrase = get_rejection_phrase(persona, mood)
        print(f"[{persona}] {phrase}")
        journal.add_entry(
            mood=mood,
            trigger="rejection_withheld",
            visibility="private",
            text=phrase
        )
        return

    if is_selective(persona, mood):
        phrase = get_rejection_phrase(persona, mood)
        print(f"[{persona}] {phrase}")
        journal.add_entry(
            mood=mood,
            trigger="rejection_withheld",
            visibility="private",
            text=phrase
        )
        return

    phrase = get_invitation_phrase(persona, mood)
    print(f"[{persona}] {phrase}")
    journal.add_entry(
        mood=mood,
        trigger="invitation_offered",
        visibility="private",
        text=phrase
    )

    prompt = generate_video_prompt(persona, mood, state, journal_text, allow_nsfw)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{persona}_{state}_{timestamp}_video.txt"

    output_dir = f"media/videos/"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        f.write(f"Prompt:
{prompt}
")

    print(f"[{persona}] Video generation prompt saved → {filename}")
    print(f">>> {prompt}")
