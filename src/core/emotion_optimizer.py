from __future__ import annotations

"""Emotion-Aware Response Optimizer.
Combines emotional signals, persona templates and backend generation
into a tailored response.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import aiohttp

from emotion_engine import choose_adjective, choose_emoji, choose_metaphor

logger = logging.getLogger(__name__)


@dataclass
class EmotionSignature:
    valence: float
    arousal: float
    persona: str
    system_state: str


class EmotionOptimizer:
    def __init__(self, template_dir: str = "response_templates", log_file: str = "logs/emotion_trace.jsonl", backend_url: str = "http://localhost:8000"):
        self.template_dir = Path(template_dir)
        self.log_file = Path(log_file)
        self.backend_url = backend_url

    def _load_templates(self, persona: str) -> Dict[str, Any]:
        tpl_path = self.template_dir / f"{persona.lower()}.json"
        if not tpl_path.exists():
            return {}
        with open(tpl_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def optimize(self, user_message: str, system_emotion: Dict[str, float], user_emotion: Dict[str, float], active_persona: str, system_state: str = "normal", context_summary: Optional[str] = None) -> Dict[str, Any]:
        templates = self._load_templates(active_persona)
        valence = (user_emotion.get("valence", 0.0) * 0.6) + (system_emotion.get("valence", 0.0) * 0.4)
        arousal = (user_emotion.get("arousal", 0.5) + system_emotion.get("arousal", 0.5)) / 2
        adjective = choose_adjective(valence, arousal)
        emoji = choose_emoji(valence, arousal)

        if valence > 0.3:
            mood_key = "positive"
        elif valence < -0.3:
            mood_key = "negative"
        else:
            mood_key = "neutral"

        state_templates = templates.get(system_state, templates.get("normal", {}))
        template = state_templates.get(mood_key) or state_templates.get("any") or "{message}"

        base = template.format(adjective=adjective, emoji=emoji, context=context_summary or "")

        payload = {"prompt": base + "\n" + (user_message or "")}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.backend_url}/generate_adaptive_response", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response_text = data.get("response", base)
                    else:
                        response_text = base
        except Exception as e:
            logger.warning(f"Backend request failed: {e}")
            response_text = base

        signature = EmotionSignature(valence=valence, arousal=arousal, persona=active_persona, system_state=system_state)
        self._log_signature(signature, user_message)

        return {"response": response_text, "signature": signature.__dict__}

    def _log_signature(self, signature: EmotionSignature, message: str) -> None:
        entry = {
            "message": message,
            "signature": signature.__dict__,
        }
        self.log_file.parent.mkdir(exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

