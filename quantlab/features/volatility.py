"""Realized volatility from daily returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(
    returns: pd.DataFrame, window: int = 63, annualize: bool = True, periods_per_year: int = 252
) -> pd.DataFrame:
    """Rolling standard deviation of returns, optionally scaled by sqrt(252)."""
    vol = returns.rolling(window, min_periods=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol
