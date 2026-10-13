import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.costs import ZERO_COST
from quantlab.data.panel import make_synthetic_panel
from quantlab.strategies.dual_momentum import DualMomentum
from quantlab.validation.sensitivity import grid_combinations, grid_sweep
from quantlab.validation.walk_forward import walk_forward

GRID = {"lookback": [60, 120], "top_n": [1, 2]}


@pytest.fixture(scope="module")
def long_panel():
    # 2010-01 .. ~2019-12, 10 years of business days
    return make_synthetic_panel(["A", "B", "C"], start="2010-01-01", periods=2600, seed=3)


def test_grid_combinations():
    combos = grid_combinations(GRID)
    assert len(combos) == 4 and {"lookback": 60, "top_n": 2} in combos


def test_grid_sweep_rows_and_columns(long_panel):
    df = grid_sweep(long_panel, DualMomentum, GRID, ZERO_COST, fixed_params={"skip": 0})
    assert len(df) == 4
    assert {"lookback", "top_n", "sharpe", "max_drawdown"} <= set(df.columns)


def test_walk_forward_windows_and_oos(long_panel):
    wf = walk_forward(
        long_panel,
        DualMomentum,
        GRID,
        ZERO_COST,
        train_years=3,
        test_years=1,
        fixed_params={"skip": 0},
        warmup_days=150,
    )
    years = sorted(set(long_panel.index.year))
    # first test year = first_year + train_years + 1 = 2014; last = 2019 => 6 windows
    assert [w.test_start.year for w in wf.windows] == list(range(2014, years[-1] + 1))
    # out-of-sample series is contiguous over the test years, no duplicates
    assert not wf.oos_returns.index.has_duplicates
    assert wf.oos_returns.index.is_monotonic_increasing
    assert wf.oos_returns.index[0].year == 2014
    for w in wf.windows:
        assert w.best_params["lookback"] in GRID["lookback"]
        assert w.best_params["top_n"] in GRID["top_n"]
        assert w.train_end < w.test_start
    assert wf.n_trials == 4 * len(wf.windows)
    assert np.isfinite(wf.oos_metrics.sharpe)
    assert len(wf.summary()) == len(wf.windows)


def test_walk_forward_params_are_json_native(long_panel):
    wf = walk_forward(
        long_panel,
        DualMomentum,
        GRID,
        ZERO_COST,
        train_years=3,
        fixed_params={"skip": 0},
        warmup_days=150,
    )
    import json

    json.dumps(wf.windows[0].best_params)


def test_walk_forward_too_short():
    short = make_synthetic_panel(["A", "B"], start="2018-01-01", periods=400)
    with pytest.raises(ValueError):
        walk_forward(short, DualMomentum, GRID, ZERO_COST, train_years=5, fixed_params={"skip": 0})


def test_oos_uses_only_test_period_returns(long_panel):
    wf = walk_forward(
        long_panel,
        DualMomentum,
        GRID,
        ZERO_COST,
        train_years=3,
        fixed_params={"skip": 0},
        warmup_days=150,
    )
    first = wf.windows[0]
    assert wf.oos_returns.loc[: first.test_end].index.min() >= pd.Timestamp("2014-01-01")
