#!/usr/bin/env python3
"""
HRM Module: Hierarchical Reflective Memory

This module formalizes memory layers: identity, beliefs, and ephemeral.
"""

import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HRMModule:
    """Hierarchical Reflective Memory Module"""

    def __init__(self):
        self.identity_layer = {}
        self.beliefs_layer = {}
        self.ephemeral_layer = {}
        self.version = "1.0.0"

    def add_to_layer(self, layer: str, key: str, value: Any):
        """Add or update an entry in the specified memory layer."""
        if layer not in ["identity", "beliefs", "ephemeral"]:
            raise ValueError(f"Invalid layer: {layer}")

        getattr(self, f"{layer}_layer")[key] = value
        logger.info(f"Added to {layer} layer: {key} -> {value}")

    def decay_memory(self, decay_rate: float = 0.1):
        """Apply decay to all layers based on the specified rate."""
        for layer_name in ["identity_layer", "beliefs_layer", "ephemeral_layer"]:
            layer = getattr(self, layer_name)
            for key in list(layer.keys()):
                if isinstance(layer[key], (int, float)):
                    layer[key] *= (1 - decay_rate)
                    if layer[key] <= 0.01:  # Threshold for removal
                        del layer[key]
                        logger.info(f"Removed {key} from {layer_name} due to decay.")

    def symbolic_link(self, source_key: str, target_key: str):
        """Create a symbolic link between two memory entries."""
        if source_key not in self.identity_layer:
            raise KeyError(f"Source key {source_key} not found in identity layer.")

        if target_key not in self.beliefs_layer:
            raise KeyError(f"Target key {target_key} not found in beliefs layer.")

        link = {"source": source_key, "target": target_key, "timestamp": time.time()}
        self.ephemeral_layer[f"link_{source_key}_{target_key}"] = link
        logger.info(f"Created symbolic link: {link}")

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the HRM system."""
        health_summary = {
            "identity_count": len(self.identity_layer),
            "beliefs_count": len(self.beliefs_layer),
            "ephemeral_count": len(self.ephemeral_layer),
            "version": self.version
        }
        logger.info(f"HRM health check: {health_summary}")
        return health_summary

    def api_safe_guard(self, operation: str):
        """Ensure the operation is safe for API access."""
        allowed_operations = ["add", "get", "decay", "link", "health_check"]
        if operation not in allowed_operations:
            raise PermissionError(f"Operation {operation} is not API-safe.")
        logger.info(f"Operation {operation} passed API-safe guard.")

if __name__ == "__main__":
    hrm = HRMModule()
    hrm.add_to_layer("identity", "name", "AI Core")
    hrm.add_to_layer("beliefs", "purpose", "Assist humans with emotional reasoning")
    hrm.add_to_layer("ephemeral", "current_task", "Processing user input")

    hrm.decay_memory()
    hrm.symbolic_link("name", "purpose")

    print(hrm.health_check())
