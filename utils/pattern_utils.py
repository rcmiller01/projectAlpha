from __future__ import annotations

import pandas as pd
from typing import List, Dict, Any


def group_memories(memories: List[Dict[str, Any]], keys: List[str]) -> Dict[str, pd.DataFrame]:
    """Group memory dicts by the given keys using pandas."""
    if not memories:
        return {}
    df = pd.DataFrame(memories)
    for key in keys:
        if key not in df.columns:
            df[key] = None
    grouped = df.groupby(keys)
    return {str(name): group for name, group in grouped}


def smooth_series(series: pd.Series, window: int = 3) -> pd.Series:
    """Return rolling mean of the series for smoothing."""
    return series.rolling(window=window, min_periods=1).mean()


def emotional_shift(values: List[float]) -> float:
    """Return shift between first and last value in a list."""
    if not values:
        return 0.0
    return values[-1] - values[0]
