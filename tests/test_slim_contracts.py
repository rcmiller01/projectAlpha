import pytest


def test_logic_high_contract_success():
    # Importing registers agents
    import slim.agents  # noqa: F401
    from slim import agents
    from slim.sdk import get_slim_status

    out = agents.logic_high("sum two numbers", {"a": 1, "b": 2})
    assert out["_contract"] == "logic_high"
    assert "reasoning" in out and "confidence" in out

    status = get_slim_status()
    assert "logic_high" in status["contracts"]
    assert "logic_high" in status["agents"]


def test_emotion_valence_input_validation_missing_required():
    import slim.agents  # noqa: F401
    from slim import agents

    # missing required 'context'
    with pytest.raises(ValueError):
        agents.emotion_valence(text="hello", context=None)  # type: ignore[arg-type]


def test_latency_timeout_enforced():
    import slim.agents  # noqa: F401
    from slim import agents

    # simulate delay beyond contract limit (logic_high max ~1500ms)
    with pytest.raises(TimeoutError):
        agents.logic_high("heavy task", {"n": 1}, simulate_ms=2000)


def test_registry_metrics_populated_after_call():
    import slim.agents  # noqa: F401
    from slim import agents
    from slim.sdk import get_slim_status

    # quick call within budget
    agents.creative_metaphor("trust", "engineers")
    status = get_slim_status()
    m = status["metrics"]["creative_metaphor"]
    assert m["total_calls"] >= 1
    # last_latency_ms may be None on very fast runs; allow either
    assert "last_ok" in m
