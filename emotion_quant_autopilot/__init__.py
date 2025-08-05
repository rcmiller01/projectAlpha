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
    from .quant_autopilot import (
        QuantizationAutopilot,
        AutopilotDatabase,
        QuantizationJob,
        AutopilotRun
    )
    from .idle_monitor import (
        IdleMonitor,
        IdleConfig,
        SystemMetrics,
        create_idle_monitor_from_config
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
    "create_idle_monitor_from_config"
]
