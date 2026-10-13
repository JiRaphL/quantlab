"""Confidence interval on the Sharpe ratio by circular block bootstrap.

Daily returns are resampled in contiguous blocks (default one month) to preserve
autocorrelation and volatility clustering, then the Sharpe is recomputed on each sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.validation.metrics import sharpe_ratio


def block_bootstrap_sharpe(
    returns: pd.Series,
    block: int = 21,
    n_boot: int = 1000,
    seed: int = 0,
    periods_per_year: int = 252,
    ci: float = 0.95,
) -> tuple[float, float, np.ndarray]:
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < block * 2:
        raise ValueError("series too short for the block size")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    samples = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        sample = r[idx[:n]]
        samples[i] = sharpe_ratio(pd.Series(sample), periods_per_year)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi), samples
