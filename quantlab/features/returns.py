"""Return transformations. Pure, causal, shape-preserving."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    """``close_t / close_{t-1} - 1``. First row is NaN."""
    return close.pct_change(fill_method=None)


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """``log(close_t / close_{t-1})``. First row is NaN."""
    return np.log(close).diff()
