"""Probabilistic and Deflated Sharpe Ratio (Bailey & López de Prado, 2012, 2014).

The question these answer: *given how many things I tried, how likely is it that the
best Sharpe I found is real rather than the luckiest draw?*

* ``probabilistic_sharpe_ratio`` is the probability that the true Sharpe exceeds a
  benchmark ``sr_benchmark``, accounting for sample length, skew and kurtosis.
* ``expected_max_sharpe`` is the Sharpe you would expect from the *best* of ``n_trials``
  strategies that all have zero true Sharpe.
* ``deflated_sharpe_ratio`` is the PSR with that expected maximum as the benchmark.
  Below ~0.95, the "discovery" is not distinguishable from selection luck.

All Sharpe ratios here are per period (daily), *not* annualised.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    if n < 2:
        raise ValueError("need n >= 2")
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1 or sr_variance <= 0:
        return 0.0
    sd = np.sqrt(sr_variance)
    a = (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * (a + b))


def deflated_sharpe_ratio(
    sr: float, n_trials: int, sr_variance: float, n: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr, benchmark, n, skew, kurtosis)


def annual_to_daily_sharpe(sr_annual: float, periods_per_year: int = 252) -> float:
    return sr_annual / np.sqrt(periods_per_year)
