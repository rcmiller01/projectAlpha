"""
Chaining planner for MoE multi-step routing.

Defines:
- Step(type, target, max_ms)
- ChainPlan(steps, budget_ms, cost_cap)
- plan_for(task, hrm_view, affect, budget_ms, cost_cap)

Also exposes simple stubs for required services:
- vector_retrieve(task, hrm_view)
- reflector_call(draft, reason, affect)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class Step:
    type: str  # "retrieve" | "reason" | "draft" | "reflect"
    target: str  # target SLiM/service name
    max_ms: int


@dataclass(frozen=True)
class ChainPlan:
    steps: list[Step]
    budget_ms: int
    cost_cap: float


def plan_for(
    task: str,
    hrm_view: dict[str, Any] | None,
    affect: dict[str, Any] | None,
    *,
    budget_ms: int = 2000,
    cost_cap: float = 1.0,
) -> ChainPlan:
    """Heuristic planner: reason→draft→reflect (+optional retrieve).

    - Include retrieve if task hints research/memory.
    - timebox each step with conservative max_ms slices of budget.
    """
    t = (task or "").lower()
    steps: list[Step] = []

    include_retrieve = any(k in t for k in ("search", "research", "remember", "recall", "context"))
    # Allocate budget slices: retrieve 20%, reason 30%, draft 35%, reflect 15%
    b_retrieve = max(100, int(budget_ms * 0.2))
    b_reason = max(150, int(budget_ms * 0.3))
    b_draft = max(200, int(budget_ms * 0.35))
    b_reflect = max(100, int(budget_ms * 0.15))

    if include_retrieve:
        steps.append(Step(type="retrieve", target="vector", max_ms=b_retrieve))

    # Choose reasoner: technical vs general
    if any(k in t for k in ("code", "algorithm", "bug", "debug", "implement", "api")):
        reason_target = "logic_high"
        draft_target = "logic_code"
    else:
        reason_target = "logic_high"
        draft_target = "creative_metaphor"  # generic creative/compose SLiM

    steps.append(Step(type="reason", target=reason_target, max_ms=b_reason))
    steps.append(Step(type="draft", target=draft_target, max_ms=b_draft))
    steps.append(Step(type="reflect", target="reflector", max_ms=b_reflect))

    return ChainPlan(steps=steps, budget_ms=budget_ms, cost_cap=cost_cap)


async def vector_retrieve(task: str, hrm_view: dict[str, Any] | None) -> dict[str, Any]:
    """Minimal stub for vector retrieval. Returns mock hits with timing.

    In real integration, this should call the vector service client.
    """
    import asyncio

    t0 = time.perf_counter()
    # Simulate small network delay
    await asyncio.sleep(0.01)
    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "hits": [
            {"id": "h1", "score": 0.78},
            {"id": "h2", "score": 0.62},
        ],
        "_elapsed_ms": elapsed,
    }


async def reflector_call(
    draft_text: str, reason_notes: str, affect: dict[str, Any] | None
) -> dict[str, Any]:
    """Minimal reflector SLiM stub.

    Returns decision "accept" unless affect['force_revise'] is truthy.
    After first revise, the caller should not ask to revise again.
    """
    import asyncio

    t0 = time.perf_counter()
    await asyncio.sleep(0.005)
    decision = "revise" if (affect or {}).get("force_revise") else "accept"
    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "decision": decision,
        "notes": "reflection complete",
        "_elapsed_ms": elapsed,
    }
