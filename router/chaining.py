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


async def execute_chain(plan: ChainPlan, context: dict) -> dict:
    """Execute a chain plan with timeboxing and accumulation of logs.

    Args:
        plan: ChainPlan with steps, budget_ms, and cost_cap
        context: Request context including hrm_view, affect, and other state

    Returns:
        dict with result, steps_log, budget_used_ms, and status information
    """
    import asyncio
    import time

    t0 = time.perf_counter()
    logs: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    revision_used = False

    # Extract required context
    hrm_view = context.get("hrm_view", {})
    affect = context.get("affect", {})
    task = context.get("task", "")

    # Budget validation
    if plan.budget_ms <= 0:
        return {
            "status": 429,
            "error": "invalid_budget",
            "plan": [s.type for s in plan.steps],
            "logs": [],
            "outputs": {},
            "_elapsed_ms": 0,
        }

    for idx, step in enumerate(plan.steps):
        now_ms = int((time.perf_counter() - t0) * 1000)
        remaining_ms = plan.budget_ms - now_ms

        if remaining_ms < 0:
            # Budget exceeded
            overage_ms = -remaining_ms
            return {
                "status": 429,
                "error": "budget_exceeded",
                "plan": [s.type for s in plan.steps],
                "budget_ms": plan.budget_ms,
                "overage_ms": overage_ms,
                "logs": logs,
                "outputs": outputs,
                "_elapsed_ms": now_ms,
            }

        # Timebox per step
        per_cap = min(step.max_ms, max(10, remaining_ms))

        # Execute step
        s_t0 = time.perf_counter()
        step_log: dict[str, Any] = {"step": step.type, "target": step.target}

        try:
            if step.type == "retrieve":
                # Execute vector retrieve with timeout
                resp = await asyncio.wait_for(
                    vector_retrieve(task, hrm_view), timeout=max(0.01, per_cap / 1000.0)
                )
                outputs["retrieve"] = resp
                elapsed_ms = resp.get("_elapsed_ms", int((time.perf_counter() - s_t0) * 1000))
                step_log.update({"_elapsed_ms": elapsed_ms})

            elif step.type in ("reason", "draft"):
                # Mock agent processing - would integrate with actual agents
                elapsed_ms = int((time.perf_counter() - s_t0) * 1000)
                mock_response = {
                    "text": f"Mock {step.type} response for: {task[:50]}...",
                    "_elapsed_ms": elapsed_ms,
                }
                outputs[step.type] = mock_response
                step_log.update({"_elapsed_ms": elapsed_ms})

            elif step.type == "reflect":
                draft_txt = (outputs.get("draft") or {}).get("text", "")
                reason_txt = (outputs.get("reason") or {}).get("text", "")

                resp = await asyncio.wait_for(
                    reflector_call(draft_txt, reason_txt, affect),
                    timeout=max(0.02, per_cap / 1000.0),
                )
                outputs["reflect"] = resp
                elapsed_ms = resp.get("_elapsed_ms", int((time.perf_counter() - s_t0) * 1000))
                step_log.update({"_elapsed_ms": elapsed_ms})

                # Handle exactly ONE revise pass as required
                if resp.get("decision") == "revise" and not revision_used:
                    revision_used = True
                    # Add one more draft with smaller slice of remaining time
                    rev_t0 = time.perf_counter()

                    # Mock revised draft
                    rev_elapsed = int((time.perf_counter() - rev_t0) * 1000)
                    outputs["draft"] = {
                        "text": f"Revised draft for: {task[:50]}...",
                        "_elapsed_ms": rev_elapsed,
                        "revision": True,
                    }
                    logs.append(
                        {
                            "step": "draft",
                            "target": "revision",
                            "_elapsed_ms": rev_elapsed,
                            "revision": True,
                        }
                    )

            else:
                step_log.update({"warning": "unknown_step"})

        except asyncio.TimeoutError:
            step_log.update({"error": "timeout", "_elapsed_ms": per_cap})
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - s_t0) * 1000)
            step_log.update({"error": str(e), "_elapsed_ms": elapsed_ms})
        finally:
            logs.append(step_log)

        # Post-step budget check
        now_ms = int((time.perf_counter() - t0) * 1000)
        if now_ms > plan.budget_ms:
            overage_ms = now_ms - plan.budget_ms
            return {
                "status": 429,
                "error": "budget_exceeded",
                "plan": [s.type for s in plan.steps],
                "budget_ms": plan.budget_ms,
                "overage_ms": overage_ms,
                "logs": logs,
                "outputs": outputs,
                "_elapsed_ms": now_ms,
            }

    elapsed_total_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "status": 200,
        "plan": [s.type for s in plan.steps],
        "logs": logs,
        "outputs": outputs,
        "_elapsed_ms": elapsed_total_ms,
        "budget_used_ms": elapsed_total_ms,
    }
