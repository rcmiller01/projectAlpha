import asyncio
import math
import time

import pytest

from backend.subagent_router import SubAgentRouter
from modules.autonomy.affect_governor import AffectGovernor, get_metrics
from modules.autonomy.drift_watchdog import DriftWatchdog


def test_governor_and_watchdog_stress(monkeypatch):
    gov = AffectGovernor(scaling=0.5, max_penalty=0.9, cooldown_seconds=1)
    wd = DriftWatchdog()

    fake_time = [0.0]

    def fake_now():
        return fake_time[0]

    monkeypatch.setattr(time, "time", fake_now)

    # 600 spikes over 60s: advance time and feed spikes
    spikes = 600
    for i in range(spikes):
        fake_time[0] = i * 0.1  # 0.1s per spike
        out = gov.apply({"source": "stress", "delta": 1.0})
        status = wd.update(1.0)
        assert out["delta_clamped"] <= out["delta"]
        assert isinstance(status["avg"], float)

    metrics = get_metrics()
    # Expect some clamping occurred
    assert metrics.get("affect_spikes_blocked_total", 0) > 0

    # Watchdog should not flap within holddown: simulate multiple breach checks within short window
    start = fake_time[0]
    count_before = wd.metrics.get("affect_watchdog_breaches_total", 0)
    for j in range(10):
        fake_time[0] = start + 0.2 * j
        wd.update(1.0)
    count_after = wd.metrics.get("affect_watchdog_breaches_total", 0)
    # only one breach counted in holddown window
    assert count_after == count_before + 1


@pytest.mark.asyncio
async def test_router_prefers_safe_or_refuses_long_form_on_breach(monkeypatch):
    # Force watchdog to breach by returning breach=True
    router = SubAgentRouter()

    orig_update = router.watchdog.update

    def fake_update(drift):
        return {"breach": True, "avg": 0.9, "window_n": 50}

    router.watchdog.update = fake_update  # type: ignore

    long_message = "x" * 400
    ctx = {"risk_mode": "high", "affect_drift": 1.0}
    resp = await router.route(long_message, ctx)

    # Response should be calming message from safe (reasoning) agent
    assert "simplify" in resp.content.lower()

    # Reset
    router.watchdog.update = orig_update  # type: ignore
