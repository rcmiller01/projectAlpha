"""Expression Dial Agent - Dynamically modulates emotional expression levels."""

from typing import Dict

# SECTION 1: Initialization & Configuration
expression_axes: Dict[str, float] = {
    "vulnerability": 0.5,
    "passion": 0.5,
    "playfulness": 0.5,
    "restraint": 0.5,
    "intimacy": 0.5,
    "responsiveness": 0.5,
}


# SECTION 2: Adjustment Methods
def set_expression_level(axis: str, value: float) -> None:
    """Set a specific expression axis to a new value."""
    if axis in expression_axes and 0.0 <= value <= 1.0:
        expression_axes[axis] = value
        log_expression_change(axis, value)
    else:
        raise ValueError("Invalid axis or out-of-range value.")


def adjust_expression_by_context(context_signal: str) -> None:
    """Modulate expression dials in response to context signals."""
    if context_signal == "user_softened":
        set_expression_level(
            "vulnerability", min(expression_axes["vulnerability"] + 0.1, 1.0)
        )
    elif context_signal == "boundary_crossing":
        set_expression_level("restraint", min(expression_axes["restraint"] + 0.2, 1.0))
    elif context_signal == "heightened_connection":
        set_expression_level("intimacy", min(expression_axes["intimacy"] + 0.15, 1.0))


# SECTION 3: Output Modulation
def modulate_output_template(template: str) -> str:
    """Alter response tone markers based on current dial settings."""
    modified_template = template

    if expression_axes["intimacy"] > 0.7:
        modified_template = modified_template.replace(
            "[tone]", "gentle and emotionally warm"
        )
    elif expression_axes["playfulness"] > 0.7:
        modified_template = modified_template.replace(
            "[tone]", "teasing and lighthearted"
        )
    elif expression_axes["restraint"] > 0.7:
        modified_template = modified_template.replace(
            "[tone]", "reserved and thoughtful"
        )

    return modified_template


# SECTION 4: Optional External Input Handling
def receive_external_adjustments(signal_dict: Dict[str, float]) -> None:
    for axis, value in signal_dict.items():
        try:
            set_expression_level(axis, float(value))
        except Exception as e:
            log_error(f"Expression adjustment failed: {e}")


# SECTION 5: Logging
def log_expression_change(axis: str, value: float) -> None:
    """Append an expression dial change entry to the log file."""
    with open("logs/expression_dial.log", "a", encoding="utf-8") as log:
        log.write(f"Changed {axis} to {value:.2f}\n")


def log_error(message: str) -> None:
    """Log errors related to expression dial operations."""
    with open("logs/expression_dial.log", "a", encoding="utf-8") as log:
        log.write(f"ERROR: {message}\n")

