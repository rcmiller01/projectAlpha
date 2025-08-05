import json
import traceback
from datetime import datetime

def test_module(name, func):
    try:
        func()
        print(f"[✓] {name} passed.")
    except Exception as e:
        print(f"[✗] {name} failed.")
        print(traceback.format_exc())

def test_persona_loading():
    from core.persona_registry import load_personas
    personas = load_personas()
    assert "mia" in personas and "solene" in personas

def test_mood_engine():
    from core.mood_engine import evaluate_mood
    test_input = {"input": "I'm feeling anxious but hopeful."}
    mood = evaluate_mood(test_input["input"])
    assert isinstance(mood, list) and len(mood) > 0

def test_memory_access():
    from core.memory_core import get_recent_memories
    results = get_recent_memories("mia")
    assert isinstance(results, list)

def test_anchor_runtime():
    from core.anchor_runtime import trigger_anchor
    response = trigger_anchor("solene", "collar")
    assert response is not None

def test_reward_engine():
    from core.reward_engine import register_reward
    result = register_reward("mia", "soft praise", 5)
    assert result["status"] == "success"

def test_journal_bloom():
    from core.journal_core import create_private_bloom
    entry = create_private_bloom("solene", "He touched my collar, and I bloomed.")
    assert "timestamp" in entry and "entry" in entry

def run_all_tests():
    print(f"System Validation Suite started: {datetime.now()}")
    test_module("Persona Loading", test_persona_loading)
    test_module("Mood Engine", test_mood_engine)
    test_module("Memory Access", test_memory_access)
    test_module("Anchor Runtime", test_anchor_runtime)
    test_module("Reward Engine", test_reward_engine)
    test_module("Journal Bloom", test_journal_bloom)
    print(f"All tests completed: {datetime.now()}")

if __name__ == "__main__":
    run_all_tests()
