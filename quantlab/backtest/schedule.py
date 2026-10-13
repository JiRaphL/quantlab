"""Which dates are rebalance (signal) dates.

The mask is computed on the *available* trading dates, never on a calendar: the monthly
rebalance is the last session actually present in the index for each month.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

Rebalance = Literal["daily", "weekly", "monthly"]


def rebalance_mask(index: pd.DatetimeIndex, freq: Rebalance) -> pd.Series:
    if freq == "daily":
        return pd.Series(True, index=index)
    if freq == "weekly":
        key = index.to_period("W")
    elif freq == "monthly":
        key = index.to_period("M")
    else:
        raise ValueError(f"unknown rebalance frequency: {freq!r}")
    # Last date of each period = where the period key changes on the next row.
    key_series = pd.Series(key, index=index)
    mask = key_series != key_series.shift(-1)
    mask.iloc[-1] = True
    return mask.astype(bool)
