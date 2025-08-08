import os

from hrm.policy_dsl import PolicyEngine, evaluate_read, evaluate_write


def policy_engine():
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, os.pardir))
    path = os.path.join(root, "hrm", "policies", "example.yaml")
    return PolicyEngine.from_yaml(path)


def test_identity_read_allowed():
    engine = policy_engine()
    d = evaluate_read(engine, layer="identity", admin_token_ok=False)
    assert d.allowed is True
    assert d.status == "ok"


def test_identity_write_needs_admin():
    engine = policy_engine()
    d = evaluate_write(engine, layer="identity", admin_token_ok=False, evidence=None)
    assert d.allowed is False
    assert d.status in {"forbidden", "needs_admin"}


def test_beliefs_write_needs_evidence():
    engine = policy_engine()
    d = evaluate_write(engine, layer="beliefs", admin_token_ok=False, evidence=None)
    assert d.allowed is False
    assert d.status == "needs_evidence"


def test_beliefs_write_with_evidence_ok():
    engine = policy_engine()
    evidence = {"source": "sensor", "confidence": 0.8}
    d = evaluate_write(engine, layer="beliefs", admin_token_ok=False, evidence=evidence)
    assert d.allowed is True


def test_ephemeral_write_ttl_propagates():
    engine = policy_engine()
    evidence = {"source": "transient", "confidence": 0.5}
    d = evaluate_write(engine, layer="ephemeral", admin_token_ok=False, evidence=evidence)
    assert d.allowed is True
    assert d.ttl_seconds == 604800
