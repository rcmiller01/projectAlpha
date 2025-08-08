"""
SLiM SDK - Specialized Language Model contracts and validation system.
Provides contract enforcement, registry management, and dry-run support.
"""

import inspect
import logging
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dry-run utilities with safe fallback for type checkers
try:
    from common.dryrun import dry_log, is_dry_run
except Exception:  # pragma: no cover - fallback stubs for type checkers
    def dry_log(*args, **kwargs):  # type: ignore[no-redef]
        return None

    def is_dry_run() -> bool:  # type: ignore[no-redef]
        return False

logger = logging.getLogger(__name__)


class SLiMContract:
    """Contract definition for SLiM agent inputs and outputs."""

    def __init__(
        self,
        name: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
        description: str = "",
        version: str = "1.0",
        side_effects: list[str] | None = None,
        max_latency_ms: Optional[float] = None,
    ):
    self.name = name
    self.input_schema = input_schema
    self.output_schema = output_schema
    self.description = description
    self.version = version
    self.side_effects = side_effects or []
    self.max_latency_ms = max_latency_ms

    def validate_input(self, data: dict[str, Any]) -> bool:
        """Validate input data against contract schema."""
        # Basic schema validation - can be enhanced with jsonschema
        return self._validate_against_schema(data, self.input_schema)

    def validate_output(self, data: dict[str, Any]) -> bool:
        """Validate output data against contract schema."""
        return self._validate_against_schema(data, self.output_schema)

    def _validate_against_schema(self, data: dict[str, Any], schema: dict[str, Any]) -> bool:
        """Basic schema validation."""
        try:
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in data:
                    logger.error(f"[SLIM_CONTRACT] Missing required field: {field}")
                    return False

            properties = schema.get("properties", {})
            for field, field_schema in properties.items():
                if field in data:
                    expected_type = field_schema.get("type")
                    if expected_type and not self._check_type(data[field], expected_type):
                        logger.error(
                            f"[SLIM_CONTRACT] Invalid type for {field}: expected {expected_type}"
                        )
                        return False

            return True
        except Exception as e:
            logger.error(f"[SLIM_CONTRACT] Schema validation error: {e}")
            return False

    """
    SLiM SDK - Specialized Language Model contracts and validation system.
    Provides contract enforcement, registry management, and dry-run support.
    """
    from __future__ import annotations

    import inspect
    import logging
    import sys
    import time
    from collections.abc import Callable
    from functools import wraps
    from pathlib import Path
    from typing import Any, Optional

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Import dry-run utilities with safe fallback
    try:
        from common.dryrun import dry_log, is_dry_run
    except Exception:  # pragma: no cover
        def dry_log(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
            return None

        def is_dry_run() -> bool:  # type: ignore[no-redef]
            return False

    logger = logging.getLogger(__name__)


    class SLiMContract:
        """Contract definition for SLiM agent inputs and outputs."""

        def __init__(
            self,
            name: str,
            input_schema: dict[str, Any],
            output_schema: dict[str, Any],
            description: str = "",
            version: str = "1.0",
            side_effects: Optional[list[str]] = None,
            max_latency_ms: Optional[float] = None,
        ) -> None:
            self.name = name
            self.input_schema = input_schema
            self.output_schema = output_schema
            self.description = description
            self.version = version
            self.side_effects = side_effects or []
            self.max_latency_ms = max_latency_ms

        def validate_input(self, data: dict[str, Any]) -> bool:
            return self._validate_against_schema(data, self.input_schema)

        def validate_output(self, data: dict[str, Any]) -> bool:
            return self._validate_against_schema(data, self.output_schema)

        def _validate_against_schema(self, data: dict[str, Any], schema: dict[str, Any]) -> bool:
            try:
                required = schema.get("required", [])
                for field in required:
                    if field not in data:
                        logger.error(f"[SLIM_CONTRACT] Missing required field: {field}")
                        return False

                props = schema.get("properties", {})
                for field, f_schema in props.items():
                    if field in data:
                        expected_type = f_schema.get("type")
                        if expected_type and not self._check_type(data[field], expected_type):
                            logger.error(
                                f"[SLIM_CONTRACT] Invalid type for {field}: expected {expected_type}"
                            )
                            return False
                return True
            except Exception as e:  # pragma: no cover
                logger.error(f"[SLIM_CONTRACT] Schema validation error: {e}")
                return False

        def _check_type(self, value: Any, expected_type: str) -> bool:
            type_mapping: dict[str, tuple[type, ...]] = {
                "string": (str,),
                "number": (int, float),
                "integer": (int,),
                "boolean": (bool,),
                "array": (list,),
                "object": (dict,),
            }
            t = type_mapping.get(expected_type)
            return isinstance(value, t) if t else True


    class SLiMRegistry:
        """Registry for SLiM contracts and agent discovery with lightweight metrics."""

        def __init__(self) -> None:
            self.contracts: dict[str, SLiMContract] = {}
            self.agents: dict[str, Callable[..., dict[str, Any]]] = {}
            self.metrics: dict[str, dict[str, Any]] = {}

        def register_contract(self, contract: SLiMContract) -> None:
            self.contracts[contract.name] = contract
            logger.info(f"[SLIM_REGISTRY] Registered contract: {contract.name} v{contract.version}")

        def register_agent(self, name: str, agent_func: Callable[..., dict[str, Any]]) -> None:
            self.agents[name] = agent_func
            if name not in self.metrics:
                self.metrics[name] = {
                    "total_calls": 0,
                    "failures": 0,
                    "last_latency_ms": None,
                    "last_error": None,
                    "last_ok": None,
                    "last_ts": None,
                }
            logger.info(f"[SLIM_REGISTRY] Registered agent: {name}")

        def get_contract(self, name: str) -> Optional[SLiMContract]:
            return self.contracts.get(name)

        def get_agent(self, name: str) -> Optional[Callable[..., dict[str, Any]]]:
            return self.agents.get(name)

        def list_contracts(self) -> list[str]:
            return list(self.contracts.keys())

        def list_agents(self) -> list[str]:
            return list(self.agents.keys())


    slim_registry = SLiMRegistry()


    def contract_validator(contract: SLiMContract):
        """Decorator for SLiM contract validation with dry-run support and metrics."""

        def decorator(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
            slim_registry.register_contract(contract)
            slim_registry.register_agent(contract.name, func)

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
                # Build input for validation
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                input_data = dict(bound.arguments)

                dry = is_dry_run()

                if not contract.validate_input(input_data):
                    raise ValueError(f"Input validation failed for contract {contract.name}")

                if dry:
                    dry_log(
                        logger,
                        f"slim.{contract.name}.execute",
                        {
                            "input_keys": list(input_data.keys()),
                            "side_effects": contract.side_effects,
                            "version": contract.version,
                        },
                    )

                try:
                    t0 = time.perf_counter()
                    out = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    if contract.max_latency_ms is not None and elapsed_ms > contract.max_latency_ms:
                        m = slim_registry.metrics.get(contract.name)
                        if m is not None:
                            m["total_calls"] += 1
                            m["failures"] += 1
                            m["last_latency_ms"] = round(elapsed_ms, 3)
                            m["last_error"] = (
                                f"Timeout: {elapsed_ms:.2f} ms > {contract.max_latency_ms:.2f} ms"
                            )
                            m["last_ok"] = False
                            m["last_ts"] = time.time()
                        raise TimeoutError(
                            f"SLiM {contract.name} exceeded max latency: {elapsed_ms:.2f} ms > {contract.max_latency_ms:.2f} ms"
                        )
                except Exception as e:
                    logger.error(f"[SLIM_CONTRACT] Execution error in {contract.name}: {e}")
                    m = slim_registry.metrics.get(contract.name)
                    if m is not None:
                        m["total_calls"] += 1
                        m["failures"] += 1
                        m["last_error"] = str(e)
                        m["last_ok"] = False
                        m["last_ts"] = time.time()
                    raise

                if not isinstance(out, dict) or not contract.validate_output(out):
                    m = slim_registry.metrics.get(contract.name)
                    if m is not None:
                        m["total_calls"] += 1
                        m["failures"] += 1
                        m["last_error"] = "Output validation failed"
                        m["last_ok"] = False
                        m["last_ts"] = time.time()
                    raise ValueError(f"Output validation failed for contract {contract.name}")

                out["_dry_run"] = dry
                out["_contract"] = contract.name
                out["_version"] = contract.version

                m = slim_registry.metrics.get(contract.name)
                if m is not None:
                    m["last_latency_ms"] = round(elapsed_ms, 3)  # type: ignore[name-defined]
                    m["total_calls"] += 1
                    m["last_error"] = None
                    m["last_ok"] = True
                    m["last_ts"] = time.time()

                if dry:
                    out["_declared_side_effects"] = contract.side_effects
                    dry_log(
                        logger,
                        f"slim.{contract.name}.completed",
                        {
                            "output_keys": list(out.keys()),
                            "side_effects_declared": len(contract.side_effects),
                        },
                    )

                return out

            return wrapper

        return decorator


    # Predefined contract schemas for common SLiM types
    LOGIC_HIGH_CONTRACT = SLiMContract(
        name="logic_high",
        input_schema={
            "type": "object",
            "required": ["reasoning_task", "context"],
            "properties": {
                "reasoning_task": {"type": "string"},
                "context": {"type": "object"},
                "complexity": {"type": "string"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["reasoning", "confidence"],
            "properties": {
                "reasoning": {"type": "string"},
                "confidence": {"type": "number"},
                "steps": {"type": "array"},
                "assumptions": {"type": "array"},
            },
        },
        description="Advanced reasoning and logical analysis",
        side_effects=["memory_access", "knowledge_retrieval"],
        max_latency_ms=1500.0,
    )

    EMOTION_VALENCE_CONTRACT = SLiMContract(
        name="emotion_valence",
        input_schema={
            "type": "object",
            "required": ["text", "context"],
            "properties": {
                "text": {"type": "string"},
                "context": {"type": "object"},
                "previous_state": {"type": "object"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["emotion", "valence", "intensity"],
            "properties": {
                "emotion": {"type": "string"},
                "valence": {"type": "number"},
                "intensity": {"type": "number"},
                "emotional_tags": {"type": "array"},
            },
        },
        description="Emotional analysis and valence detection",
        side_effects=["hrm_ephemeral_write"],
        max_latency_ms=800.0,
    )

    CREATIVE_METAPHOR_CONTRACT = SLiMContract(
        name="creative_metaphor",
        input_schema={
            "type": "object",
            "required": ["concept", "target_audience"],
            "properties": {
                "concept": {"type": "string"},
                "target_audience": {"type": "string"},
                "style": {"type": "string"},
                "complexity": {"type": "string"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["metaphor", "explanation", "effectiveness"],
            "properties": {
                "metaphor": {"type": "string"},
                "explanation": {"type": "string"},
                "effectiveness": {"type": "number"},
                "alternative_metaphors": {"type": "array"},
            },
        },
        description="Creative metaphor generation and explanation",
        side_effects=["knowledge_synthesis"],
        max_latency_ms=1200.0,
    )


    def get_slim_status() -> dict[str, Any]:
        metrics_snapshot: dict[str, Any] = {}
        for name, m in slim_registry.metrics.items():
            metrics_snapshot[name] = {
                "total_calls": m.get("total_calls", 0),
                "failures": m.get("failures", 0),
                "last_latency_ms": m.get("last_latency_ms"),
                "last_ok": m.get("last_ok"),
                "last_ts": m.get("last_ts"),
            }

        return {
            "registry_size": len(slim_registry.contracts),
            "contracts": slim_registry.list_contracts(),
            "agents": slim_registry.list_agents(),
            "metrics": metrics_snapshot,
            "dry_run_mode": is_dry_run(),
        }


    def validate_slim_ecosystem() -> dict[str, Any]:
        issues: list[str] = []
        for contract_name in slim_registry.contracts:
            if contract_name not in slim_registry.agents:
                issues.append(f"Contract {contract_name} has no registered agent")
        for agent_name in slim_registry.agents:
            if agent_name not in slim_registry.contracts:
                issues.append(f"Agent {agent_name} has no contract definition")
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_contracts": len(slim_registry.contracts),
            "total_agents": len(slim_registry.agents),
        }
