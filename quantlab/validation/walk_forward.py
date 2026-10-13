"""Anchored walk-forward validation.

For each test year, parameters are chosen on the preceding ``train_years`` of data only
(by the best in-sample ``score`` over the grid), then applied to the test year. The
concatenated test slices form a genuinely out-of-sample track record: every decision
was taken without seeing the data it is evaluated on.

``n_trials`` counts every backtest run during selection: this feeds the deflated Sharpe
ratio so the search itself is accounted for.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import run_backtest
from quantlab.backtest.schedule import Rebalance
from quantlab.data.panel import PricePanel
from quantlab.strategies.base import Strategy
from quantlab.validation.metrics import Metrics, compute_metrics
from quantlab.validation.sensitivity import grid_combinations, grid_sweep


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_score: float
    test_metrics: Metrics


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    oos_returns: pd.Series
    oos_turnover: pd.Series
    oos_costs: pd.Series
    oos_metrics: Metrics
    n_trials: int
    grid: dict[str, list]
    score: str

    def summary(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            rows.append(
                {
                    "test_start": w.test_start.date(),
                    "test_end": w.test_end.date(),
                    **{f"param_{k}": v for k, v in w.best_params.items()},
                    "train_score": w.train_score,
                    "test_sharpe": w.test_metrics.sharpe,
                    "test_cagr": w.test_metrics.cagr,
                    "test_max_dd": w.test_metrics.max_drawdown,
                }
            )
        return pd.DataFrame(rows)


def walk_forward(
    panel: PricePanel,
    strategy_cls: type[Strategy],
    grid: dict[str, list],
    cost_model: CostModel,
    rebalance: Rebalance = "monthly",
    train_years: int = 5,
    test_years: int = 1,
    score: str = "sharpe",
    fixed_params: dict | None = None,
    warmup_days: int = 300,
) -> WalkForwardResult:
    years = sorted(set(panel.index.year))
    first_year = years[0]
    # First test year: enough history for warm-up plus the training window.
    first_test_year = first_year + train_years + 1
    windows: list[WalkForwardWindow] = []
    oos_parts: list[pd.Series] = []
    oos_to: list[pd.Series] = []
    oos_co: list[pd.Series] = []
    combos = grid_combinations(grid)
    n_trials = 0

    test_year = first_test_year
    while test_year <= years[-1]:
        test_start = pd.Timestamp(year=test_year, month=1, day=1)
        test_end = pd.Timestamp(year=test_year + test_years - 1, month=12, day=31)
        train_start = pd.Timestamp(year=test_year - train_years, month=1, day=1)
        train_end = test_start - pd.Timedelta(days=1)
        if test_start > panel.index[-1]:
            break

        train_panel = panel.slice(train_start - pd.Timedelta(days=warmup_days * 2), train_end)
        sweep = grid_sweep(
            train_panel,
            strategy_cls,
            grid,
            cost_model,
            rebalance,
            fixed_params=fixed_params,
            score_start=train_start,
            score_end=train_end,
            combos=combos,
        )
        n_trials += len(sweep)
        best_row = sweep.sort_values(score, ascending=False).iloc[0]
        best_params = {k: _native(best_row[k]) for k in grid}

        test_panel = panel.slice(test_start - pd.Timedelta(days=warmup_days * 2), test_end)
        strategy = strategy_cls(**(fixed_params or {}), **best_params)
        result = run_backtest(
            test_panel, strategy.target_weights(test_panel), cost_model, rebalance
        )
        test_slice = result.slice(test_start, test_end)
        test_metrics = compute_metrics(
            test_slice.returns, turnover=test_slice.turnover, costs=test_slice.costs
        )
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_slice.index[0],
                test_end=test_slice.index[-1],
                best_params=best_params,
                train_score=float(best_row[score]),
                test_metrics=test_metrics,
            )
        )
        oos_parts.append(test_slice.returns)
        oos_to.append(test_slice.turnover)
        oos_co.append(test_slice.costs)
        test_year += test_years

    if not windows:
        raise ValueError("not enough history for a single walk-forward window")
    oos_returns = pd.concat(oos_parts)
    oos_turnover = pd.concat(oos_to)
    oos_costs = pd.concat(oos_co)
    return WalkForwardResult(
        windows=windows,
        oos_returns=oos_returns,
        oos_turnover=oos_turnover,
        oos_costs=oos_costs,
        oos_metrics=compute_metrics(oos_returns, turnover=oos_turnover, costs=oos_costs),
        n_trials=n_trials,
        grid=grid,
        score=score,
    )


def _native(value):
    """numpy scalar -> python scalar (so params serialise cleanly)."""
    return value.item() if hasattr(value, "item") else value
