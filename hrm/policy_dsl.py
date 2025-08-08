from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import yaml

Layer = Literal["identity", "beliefs", "ephemeral"]
Action = Literal["read", "write", "delete"]


@dataclass
class PolicyRule:
    layer: Layer
    action: Action
    allow: bool
    requires_admin: bool = False
    requires_evidence: bool = False
    ttl_seconds: Optional[int] = None  # for ephemeral writes
    reason: str = ""


@dataclass
class PolicyDecision:
    allowed: bool
    status: Literal["ok", "forbidden", "needs_admin", "needs_evidence"]
    reason: str
    ttl_seconds: Optional[int] = None


class PolicyEngine:
    """
    Evaluate HRM access decisions using a tiny declarative policy DSL loaded from YAML.
    """

    def __init__(self, rules: list[PolicyRule]):
        self._rules = rules

    @classmethod
    def from_yaml(cls, path: str) -> PolicyEngine:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rules: list[PolicyRule] = []
        for item in data.get("rules", []):
            rules.append(
                PolicyRule(
                    layer=item["layer"],
                    action=item["action"],
                    allow=bool(item.get("allow", False)),
                    requires_admin=bool(item.get("requires_admin", False)),
                    requires_evidence=bool(item.get("requires_evidence", False)),
                    ttl_seconds=item.get("ttl_seconds"),
                    reason=item.get("reason", ""),
                )
            )
        return cls(rules)

    def decide(
        self,
        *,
        layer: Layer,
        action: Action,
        is_admin: bool,
        has_evidence: bool,
    ) -> PolicyDecision:
        # First matching rule wins; fallback = deny
        for r in self._rules:
            if r.layer == layer and r.action == action:
                if not r.allow:
                    return PolicyDecision(False, "forbidden", r.reason or "Denied by policy.")
                if r.requires_admin and not is_admin:
                    return PolicyDecision(False, "needs_admin", r.reason or "Admin required.")
                if r.requires_evidence and not has_evidence:
                    return PolicyDecision(False, "needs_evidence", r.reason or "Evidence required.")
                return PolicyDecision(True, "ok", r.reason or "Allowed.", ttl_seconds=r.ttl_seconds)
        return PolicyDecision(False, "forbidden", "No matching rule.")


# --- Convenience helpers for HRM APIs ---------------------------------------


def evaluate_write(
    engine: PolicyEngine,
    *,
    layer: Layer,
    admin_token_ok: bool,
    evidence: Optional[dict[str, Any]] = None,
) -> PolicyDecision:
    has_evidence = bool(
        evidence and evidence.get("source") and evidence.get("confidence") is not None
    )
    return engine.decide(
        layer=layer, action="write", is_admin=admin_token_ok, has_evidence=has_evidence
    )


def evaluate_read(
    engine: PolicyEngine,
    *,
    layer: Layer,
    admin_token_ok: bool,
) -> PolicyDecision:
    # Reads rarely need evidence; admin may see more fields
    return engine.decide(layer=layer, action="read", is_admin=admin_token_ok, has_evidence=True)


# TODO: Optional: cache decisions; add hot-reload on file change timestamp.
