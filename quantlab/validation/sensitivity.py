"""Parameter grid sweep: one backtest per combination, one row of metrics each.

A robust strategy works on a broad region of its parameter space. If only a single point
of the grid looks good, that point was found by luck.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import run_backtest
from quantlab.backtest.schedule import Rebalance
from quantlab.data.panel import PricePanel
from quantlab.strategies.base import Strategy
from quantlab.validation.metrics import compute_metrics


def grid_combinations(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def grid_sweep(
    panel: PricePanel,
    strategy_cls: type[Strategy],
    grid: dict[str, list],
    cost_model: CostModel,
    rebalance: Rebalance = "monthly",
    fixed_params: dict | None = None,
    score_start: pd.Timestamp | None = None,
    score_end: pd.Timestamp | None = None,
    combos: Iterable[dict] | None = None,
) -> pd.DataFrame:
    """Run every combination of ``grid`` (merged with ``fixed_params``) and return a
    DataFrame with one row per combination: the parameters, then the metrics.

    ``score_start``/``score_end`` restrict the period on which metrics are computed
    (the backtest itself runs on the whole panel so warm-up history is available)."""
    rows = []
    fixed = fixed_params or {}
    for params in combos if combos is not None else grid_combinations(grid):
        strategy = strategy_cls(**fixed, **params)
        result = run_backtest(panel, strategy.target_weights(panel), cost_model, rebalance)
        scored = result.slice(score_start, score_end)
        metrics = compute_metrics(scored.returns, turnover=scored.turnover, costs=scored.costs)
        rows.append({**params, **metrics.as_dict()})
    return pd.DataFrame(rows)
