"""Price momentum: the trailing return over ``lookback`` sessions, skipping the most
recent ``skip`` sessions (the classic "12-1" momentum skips the last month because of
short-term reversal). Causal by construction: only ``shift`` with positive lags."""

from __future__ import annotations

import pandas as pd


def momentum(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """``close_{t-skip} / close_{t-lookback} - 1``. NaN for the first ``lookback`` rows."""
    if skip < 0 or lookback <= skip:
        raise ValueError("need 0 <= skip < lookback")
    return close.shift(skip) / close.shift(lookback) - 1.0
