from datetime import datetime
from typing import List, Optional, Dict, Any
import hashlib
from pydantic import BaseModel, Field
import uuid
from textblob import TextBlob

class Emotion(BaseModel):
    valence: float = Field(0.0, description="-1 to 1 sentiment valence")
    arousal: float = Field(0.0, description="0 to 1 intensity")

class SelfReport(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    persona: Optional[str] = None
    emotion: Emotion = Field(default_factory=Emotion)
    motivation: List[str] = Field(default_factory=list)
    decision_factors: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""
    session_hash: Optional[str] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


def create_self_report(
    last_output: str,
    memory_system: Optional[Any] = None,
    sentiment_analysis: Optional[Any] = None,
    persona_manager: Optional[Any] = None,
    reflection_engine: Optional[Any] = None,
    response_context: Optional[Any] = None,
) -> SelfReport:
    """Generate a SelfReport using available integration points."""

    persona_name = None
    if persona_manager and hasattr(persona_manager, "get_active_persona"):
        try:
            persona = persona_manager.get_active_persona()
            if isinstance(persona, dict):
                persona_name = persona.get("name") or persona.get("id")
            elif persona is not None:
                persona_name = getattr(persona, "name", str(persona))
        except Exception:
            pass

    valence = 0.0
    arousal = 0.0
    if sentiment_analysis and hasattr(sentiment_analysis, "get_current_state"):
        try:
            state = sentiment_analysis.get_current_state()
            valence = float(state.get("valence", 0.0))
            arousal = float(state.get("arousal", 0.0))
        except Exception:
            pass
    else:
        try:
            blob = TextBlob(last_output)
            valence = blob.sentiment.polarity
            arousal = abs(blob.sentiment.subjectivity)
        except Exception:
            pass

    motivation = []
    if reflection_engine and hasattr(reflection_engine, "get_last_insight"):
        try:
            insight = reflection_engine.get_last_insight()
            if insight:
                motivation.extend(insight.get("tags", []))
        except Exception:
            pass

    decision_factors = []
    session_hash = None
    if memory_system and hasattr(memory_system, "get_last_memory_session"):
        try:
            session = memory_system.get_last_memory_session()
            if session:
                trend = session.get("sentiment_trend")
                if trend is not None:
                    decision_factors.append(f"recent sentiment {trend:+.2f}")
                sid = session.get("session_id")
                if sid:
                    session_hash = hashlib.sha256(sid.encode()).hexdigest()[:8]
        except Exception:
            pass

    if response_context and hasattr(response_context, "get_last_response_metadata"):
        try:
            meta = response_context.get_last_response_metadata()
            if meta:
                reason = meta.get("reason") or meta.get("handler")
                if reason:
                    decision_factors.append(reason)
        except Exception:
            pass

    summary_parts = []
    if persona_name:
        summary_parts.append(f"Persona {persona_name} active.")
    summary_parts.append(f"Valence {valence:+.2f}, Arousal {arousal:+.2f}.")
    if motivation:
        summary_parts.append("Motivation: " + ", ".join(motivation))
    if decision_factors:
        summary_parts.append("Factors: " + ", ".join(decision_factors))

    summary = " " .join(summary_parts)

    return SelfReport(
        persona=persona_name,
        emotion=Emotion(valence=valence, arousal=arousal),
        motivation=motivation,
        decision_factors=decision_factors,
        session_hash=session_hash,
        confidence=0.75,
        summary=summary,
    )
