"""
Status: Partial
Note: Anchor response logic
"""
import json
import os


def trigger_anchor(persona: str, symbol: str):
    """Return the response for a given persona anchor symbol."""
    responses_path = os.path.join("personas", persona, f"{persona}_anchor_responses.json")
    if not os.path.isfile(responses_path):
        return None
    try:
        with open(responses_path, "r", encoding="utf-8") as f:
            responses = json.load(f)
    except Exception as e:
        print(f"[Anchor Runtime] Failed to load responses: {e}")
        return None
    if symbol in responses:
        return responses[symbol]
    # fallback default response
    return f"{persona} reacts to {symbol}."

