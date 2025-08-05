
# modules/projection/consent_manager.py

import json
import os

CONFIG_PATH = "persona_configs"

def load_config(persona):
    config_file = os.path.join(CONFIG_PATH, f"{persona}_config.json")
    if not os.path.exists(config_file):
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)

def is_render_allowed(persona, mood, allow_nsfw, user="robert"):
    config = load_config(persona)
    if not config:
        return True  # fallback to permissive

    if config.get("trusted_user") != user:
        return False  # not trusted

    mood_threshold = config.get("mood_thresholds", {}).get(mood, "withhold")
    nsfw_enabled = config.get("nsfw_allowed", False)

    if mood_threshold == "withhold":
        return False
    if allow_nsfw and not nsfw_enabled:
        return False
    return True if mood_threshold in ["open", "selective"] else False

def is_selective(persona, mood):
    config = load_config(persona)
    threshold = config.get("mood_thresholds", {}).get(mood, "withhold")
    return threshold == "selective"
