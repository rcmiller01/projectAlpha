"""
SLiM SDK - Specialized Language Model contracts and validation system.
Provides contract enforcement, registry management, and dry-run support.
"""

import inspect
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, Optional

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

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if expected_type in type_mapping:
            return isinstance(value, type_mapping[expected_type])
        return True  # Allow unknown types for now


class SLiMRegistry:
    """Registry for SLiM contracts and agent discovery."""

    def __init__(self):
        self.contracts: dict[str, SLiMContract] = {}
        self.agents: dict[str, Callable] = {}
        self.metrics: dict[str, dict[str, Any]] = {}

    def register_contract(self, contract: SLiMContract) -> None:
        """Register a SLiM contract."""
        self.contracts[contract.name] = contract
        # Initialize metrics for this contract
        self.metrics[contract.name] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "timeout_errors": 0,
            "validation_errors": 0,
        }
        logger.info(f"[SLIM_REGISTRY] Registered contract: {contract.name} v{contract.version}")

    def register_agent(self, name: str, agent_func: Callable) -> None:
        """Register a SLiM agent function."""
        self.agents[name] = agent_func
        logger.info(f"[SLIM_REGISTRY] Registered agent: {name}")

    def get_contract(self, name: str) -> Optional[SLiMContract]:
        """Get contract by name."""
        return self.contracts.get(name)

    def get_agent(self, name: str) -> Optional[Callable]:
        """Get agent function by name."""
        return self.agents.get(name)

    def list_contracts(self) -> list[str]:
        """List all registered contracts."""
        return list(self.contracts.keys())

    def list_agents(self) -> list[str]:
        """List all registered agents."""
        return list(self.agents.keys())

    def update_metrics(
        self,
        contract_name: str,
        success: bool,
        latency_ms: float,
        error_type: Optional[str] = None,
    ) -> None:
        """Update metrics for a contract execution."""
        if contract_name not in self.metrics:
            return

        metrics = self.metrics[contract_name]
        metrics["total_calls"] += 1
        metrics["total_latency_ms"] += latency_ms

        if success:
            metrics["successful_calls"] += 1
        else:
            metrics["failed_calls"] += 1
            if error_type == "timeout":
                metrics["timeout_errors"] += 1
            elif error_type == "validation":
                metrics["validation_errors"] += 1

        # Update average latency
        if metrics["total_calls"] > 0:
            metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["total_calls"]

    def get_metrics(self, contract_name: str) -> dict[str, Any]:
        """Get metrics for a specific contract."""
        return self.metrics.get(contract_name, {})

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get all contract metrics."""
        return self.metrics.copy()


# Global registry instance
slim_registry = SLiMRegistry()


def contract_validator(contract: SLiMContract):
    """Decorator for SLiM contract validation with dry-run support."""

    def decorator(func: Callable) -> Callable:
        # Register the agent and contract
        slim_registry.register_contract(contract)
        slim_registry.register_agent(contract.name, func)

        @wraps(func)
        def wrapper(*args, **kwargs) -> dict[str, Any]:
            start_time = time.time()
            error_type = None
            success = False

            try:
                # Convert args/kwargs to input dict for validation
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                input_data = dict(bound_args.arguments)

                # Check dry-run mode
                dry = is_dry_run()

                # Validate input against contract
                if not contract.validate_input(input_data):
                    error_type = "validation"
                    raise ValueError(f"Input validation failed for contract {contract.name}")

                # Log dry-run behavior
                if dry:
                    dry_log(
                        logger,
                        f"slim.{contract.name}.execute",
                        {
                            "input_keys": list(input_data.keys()),
                            "side_effects": contract.side_effects,
                            "version": contract.version,
                            "max_latency_ms": contract.max_latency_ms,
                        },
                    )

                # Execute the function
                try:
                    out: dict[str, Any] = func(*args, **kwargs)
                except Exception as e:
                    error_type = "execution"
                    logger.error(f"[SLIM_CONTRACT] Execution error in {contract.name}: {e}")
                    raise

                # Check latency if specified
                elapsed_ms = (time.time() - start_time) * 1000
                if contract.max_latency_ms and elapsed_ms > contract.max_latency_ms:
                    error_type = "timeout"
                    raise TimeoutError(
                        f"Contract {contract.name} exceeded max latency: "
                        f"{elapsed_ms:.1f}ms > {contract.max_latency_ms}ms"
                    )

                # Validate output against contract
                if not contract.validate_output(out):
                    error_type = "validation"
                    raise ValueError(f"Output validation failed for contract {contract.name}")

                # Add dry-run metadata to output
                out["_dry_run"] = dry
                out["_contract"] = contract.name
                out["_version"] = contract.version
                out["_latency_ms"] = elapsed_ms

                if dry:
                    out["_declared_side_effects"] = contract.side_effects
                    dry_log(
                        logger,
                        f"slim.{contract.name}.completed",
                        {
                            "output_keys": list(out.keys()),
                            "side_effects_declared": len(contract.side_effects),
                            "latency_ms": elapsed_ms,
                        },
                    )

                success = True
                return out

            finally:
                # Update metrics regardless of success/failure
                elapsed_ms = (time.time() - start_time) * 1000
                slim_registry.update_metrics(contract.name, success, elapsed_ms, error_type)

        # Attach contract to function for introspection
        wrapper._slim_contract = contract
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
            "complexity": {"type": "string", "enum": ["simple", "moderate", "complex"]},
        },
    },
    output_schema={
        "type": "object",
        "required": ["reasoning", "confidence"],
        "properties": {
            "reasoning": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "steps": {"type": "array"},
            "assumptions": {"type": "array"},
        },
    },
    description="Advanced reasoning and logical analysis",
    side_effects=["memory_access", "knowledge_retrieval"],
    max_latency_ms=5000.0,
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
            "valence": {"type": "number", "minimum": -1, "maximum": 1},
            "intensity": {"type": "number", "minimum": 0, "maximum": 1},
            "emotional_tags": {"type": "array"},
        },
    },
    description="Emotional analysis and valence detection",
    side_effects=["hrm_ephemeral_write"],
    max_latency_ms=3000.0,
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
            "effectiveness": {"type": "number", "minimum": 0, "maximum": 1},
            "alternative_metaphors": {"type": "array"},
        },
    },
    description="Creative metaphor generation and explanation",
    side_effects=["knowledge_synthesis"],
    max_latency_ms=4000.0,
)


def get_slim_status() -> dict[str, Any]:
    """Get current SLiM registry status for monitoring."""
    return {
        "registry_size": len(slim_registry.contracts),
        "contracts": slim_registry.list_contracts(),
        "agents": slim_registry.list_agents(),
        "dry_run_mode": is_dry_run(),
        "metrics": slim_registry.get_all_metrics(),
    }


def validate_slim_ecosystem() -> dict[str, Any]:
    """Validate the SLiM ecosystem configuration."""
    issues = []

    # Check for contract-agent mismatches
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
        "registry_status": get_slim_status(),
    }


# Export list for public API
__all__ = [
    "SLiMContract",
    "SLiMRegistry",
    "slim_registry",
    "contract_validator",
    "LOGIC_HIGH_CONTRACT",
    "EMOTION_VALENCE_CONTRACT",
    "CREATIVE_METAPHOR_CONTRACT",
    "get_slim_status",
    "validate_slim_ecosystem",
]
