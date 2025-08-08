import asyncio

import pytest

from backend.subagent_router import SubAgentRouter
from router.chaining import ChainPlan, Step, plan_for


@pytest.mark.asyncio
async def test_chain_happy_path():
    router = SubAgentRouter()
    ctx = {"budget_ms": 1500}
    res = await router.route_chain("explain quicksort algorithm", ctx)
    assert res["status"] == 200
    # Ensure steps executed in order, including reflect
    assert "reflect" in res["plan"]
    # Logs are ranked by occurrence; ensure present and have elapsed
    assert isinstance(res["logs"], list) and len(res["logs"]) >= 2
    assert any("_elapsed_ms" in l for l in res["logs"])


@pytest.mark.asyncio
async def test_chain_budget_breach():
    router = SubAgentRouter()
    # Force extremely low budget to trigger 429
    ctx = {"budget_ms": 1}
    res = await router.route_chain("do a long report", ctx)
    assert res["status"] == 429
    assert res["error"] in ("invalid_budget", "budget_exceeded")


@pytest.mark.asyncio
async def test_chain_reflect_triggers_single_revise():
    router = SubAgentRouter()
    # force reflect to request revise once
    ctx = {"budget_ms": 1200, "affect_valence": 0.1, "force_revise": True}
    res = await router.route_chain("compose a short note", ctx)
    assert res["status"] == 200
    # Ensure exactly one revision draft was added
    drafts = [l for l in res["logs"] if l.get("step") == "draft"]
    assert any(l.get("revision") is True for l in drafts)
    assert sum(1 for l in drafts if l.get("revision") is True) == 1
