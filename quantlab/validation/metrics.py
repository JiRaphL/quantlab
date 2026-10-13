"""Performance metrics from a series of daily returns.

Everything is computed from returns, never from a NAV level, so metrics are independent
of the initial capital. Sharpe uses the arithmetic mean of excess daily returns scaled
by sqrt(252): the convention of the deflated-Sharpe literature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Metrics:
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar: float
    skew: float
    kurtosis: float  # non-excess (normal = 3)
    annual_turnover: float | None
    annual_cost_bps: float | None
    n_days: int
    total_return: float

    def as_dict(self) -> dict:
        return asdict(self)


def drawdown_series(returns: pd.Series) -> pd.Series:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    peak = nav.cummax()
    return nav / peak - 1.0


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch (in sessions) between a peak and the next recovery to that peak."""
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    peak = nav.cummax()
    under = (nav < peak).to_numpy()
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def compute_metrics(
    returns: pd.Series,
    periods_per_year: int = 252,
    rf: float = 0.0,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> Metrics:
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 2:
        raise ValueError("need at least two returns")
    years = n / periods_per_year
    total = float((1.0 + r).prod() - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1.0 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))

    downside = r[r < 0]
    down_sd = float(np.sqrt((downside**2).sum() / n) * np.sqrt(periods_per_year)) if n else 0.0
    mean_ann = float(r.mean() * periods_per_year - rf)
    sortino = mean_ann / down_sd if down_sd > 0 else (float("inf") if mean_ann > 0 else 0.0)

    mdd = float(drawdown_series(r).min())
    calmar = cagr / abs(mdd) if mdd < 0 else (float("inf") if cagr > 0 else 0.0)

    skew = float(stats.skew(r, bias=False)) if vol > 0 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False, bias=False)) if vol > 0 else 3.0

    annual_turnover = float(turnover.sum() / years) if turnover is not None else None
    annual_cost_bps = float(costs.sum() / years * 1e4) if costs is not None else None

    return Metrics(
        cagr=cagr,
        volatility=vol,
        sharpe=sharpe_ratio(r, periods_per_year, rf),
        sortino=float(sortino),
        max_drawdown=mdd,
        max_drawdown_duration_days=max_drawdown_duration(r),
        calmar=float(calmar),
        skew=skew,
        kurtosis=kurt,
        annual_turnover=annual_turnover,
        annual_cost_bps=annual_cost_bps,
        n_days=n,
        total_return=total,
    )
