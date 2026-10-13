"""Output of a backtest run. Everything downstream (metrics, reports, registry) reads
this and nothing else."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    nav: pd.Series  # net asset value after costs
    nav_gross: pd.Series  # NAV ignoring costs
    returns: pd.Series  # daily net returns (first value 0)
    returns_gross: pd.Series
    weights: pd.DataFrame  # post-trade weights held at each close, risky assets only
    turnover: pd.Series  # one-way turnover (sum |delta w|) on execution dates, 0 elsewhere
    costs: pd.Series  # cost paid as a fraction of NAV, per date
    trades: pd.DataFrame  # date, symbol, delta_weight, delta_value, cost
    config: dict = field(default_factory=dict)

    @property
    def cash(self) -> pd.Series:
        return 1.0 - self.weights.sum(axis=1)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.nav.index

    def slice(self, start=None, end=None) -> BacktestResult:
        """Restrict to a date range. NAV is rebased so the first value equals the
        original NAV at that date (returns are untouched)."""
        sl = slice(start, end)
        return BacktestResult(
            nav=self.nav.loc[sl],
            nav_gross=self.nav_gross.loc[sl],
            returns=self.returns.loc[sl],
            returns_gross=self.returns_gross.loc[sl],
            weights=self.weights.loc[sl],
            turnover=self.turnover.loc[sl],
            costs=self.costs.loc[sl],
            trades=self.trades[
                (self.trades["date"] >= (start or self.trades["date"].min()))
                & (self.trades["date"] <= (end or self.trades["date"].max()))
            ]
            if len(self.trades)
            else self.trades,
            config=dict(self.config),
        )
