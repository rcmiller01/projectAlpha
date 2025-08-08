"""
SLiM SDK - Specialized Language Model contracts and validation system.
Provides contract enforcement, registry management, and dry-run support.
"""

import inspect
import json
import logging
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dry-run utilities
from common.dryrun import dry_log, is_dry_run

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
    ):
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.description = description
        self.version = version
        self.side_effects = side_effects or []

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

    def register_contract(self, contract: SLiMContract) -> None:
        """Register a SLiM contract."""
        self.contracts[contract.name] = contract
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
            # Convert args/kwargs to input dict for validation
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            input_data = dict(bound_args.arguments)

            # Check dry-run mode
            dry = is_dry_run()

            # Validate input against contract
            if not contract.validate_input(input_data):
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
                    },
                )
                # Skip heavy/external operations inside SLiMs if they would be called internally
                # but still execute pure compute to get output shape if cheap.
                # For now, we'll still execute but mark as dry-run

            # Execute the function
            try:
                out: dict[str, Any] = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[SLIM_CONTRACT] Execution error in {contract.name}: {e}")
                raise

            # Validate output against contract
            if not contract.validate_output(out):
                raise ValueError(f"Output validation failed for contract {contract.name}")

            # Add dry-run metadata to output
            out["_dry_run"] = dry
            out["_contract"] = contract.name
            out["_version"] = contract.version

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
)


def get_slim_status() -> dict[str, Any]:
    """Get current SLiM registry status for monitoring."""
    return {
        "registry_size": len(slim_registry.contracts),
        "contracts": slim_registry.list_contracts(),
        "agents": slim_registry.list_agents(),
        "dry_run_mode": is_dry_run(),
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
    }
