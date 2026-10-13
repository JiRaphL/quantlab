"""Trend filters based on simple moving averages."""

from __future__ import annotations

import pandas as pd


def sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return close.rolling(window, min_periods=window).mean()


def above_sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """True where price is above its SMA. False while the SMA is undefined."""
    return (close > sma(close, window)).fillna(False).astype(bool)
