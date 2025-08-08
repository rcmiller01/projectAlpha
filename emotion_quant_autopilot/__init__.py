"""
Emotion Quantization Autopilot Package
Fully autonomous emotional quantization system for local LLMs

Components:
- quant_autopilot: Main autopilot controller
- idle_monitor: System idle state monitoring
- setup_autopilot: Installation and setup utilities

Usage:
    from emotion_quant_autopilot import QuantizationAutopilot, IdleMonitor

    autopilot = QuantizationAutopilot("autopilot_config.json")
    autopilot.start()
"""

__version__ = "1.0.0"
__author__ = "Dolphin AI Team"
__description__ = "Autonomous emotional quantization system for local LLMs"

# Import main classes for easy access
try:
    from .idle_monitor import (
        IdleConfig,
        IdleMonitor,
        SystemMetrics,
        create_idle_monitor_from_config,
    )
    from .quant_autopilot import (
        AutopilotDatabase,
        AutopilotRun,
        QuantizationAutopilot,
        QuantizationJob,
    )
except ImportError:
    # Handle relative imports when run as script
    pass

__all__ = [
    "QuantizationAutopilot",
    "AutopilotDatabase",
    "QuantizationJob",
    "AutopilotRun",
    "IdleMonitor",
    "IdleConfig",
    "SystemMetrics",
    "create_idle_monitor_from_config",
]
