"""Anchor and Centering System - Emotional Regulation and Safety Net."""

EMOTIONAL_BOUNDARIES = {
    "playfulness": (0.1, 0.9),
    "vulnerability": (0.0, 0.85),
    "intimacy": (0.0, 0.95),
    "responsiveness": (0.2, 0.95),
    "restraint": (0.05, 0.95),
    "sensuality": (0.0, 0.9),
}

ANCHOR_POINT = {
    "playfulness": 0.5,
    "vulnerability": 0.4,
    "intimacy": 0.5,
    "responsiveness": 0.6,
    "restraint": 0.5,
    "sensuality": 0.4,
}


def clamp_to_bounds(expression_dial: dict) -> None:
    """Clamp emotional expression values within defined boundaries."""
    for axis, (low, high) in EMOTIONAL_BOUNDARIES.items():
        if axis in expression_dial:
            if expression_dial[axis] < low:
                expression_dial[axis] = low
            elif expression_dial[axis] > high:
                expression_dial[axis] = high


def emergency_center(expression_dial: dict, intensity: float = 1.0) -> None:
    """Pull emotional axes toward their anchor points."""
    for axis, anchor in ANCHOR_POINT.items():
        if axis in expression_dial:
            current = expression_dial[axis]
            expression_dial[axis] = (1 - intensity) * current + intensity * anchor


def emotional_watchdog(expression_dial: dict, threshold: float = 0.95) -> None:
    """Scan for out-of-range values and trigger centering if needed."""
    for axis, value in expression_dial.items():
        low, high = EMOTIONAL_BOUNDARIES.get(axis, (0.0, 1.0))
        if value < low or value > high:
            log_anchor_trigger(axis, value)
            emergency_center(expression_dial, intensity=0.8)
            break


def log_anchor_trigger(axis: str, value: float) -> None:
    """Record an anchor trigger event."""
    try:
        with open("logs/anchor_trigger.log", "a", encoding="utf-8") as log:
            log.write(f"Axis {axis} out of range with value {value:.2f} - anchor activated.\n")
    except Exception:
        pass
