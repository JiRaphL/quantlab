import numpy as np
import pandas as pd
import pytest

from quantlab.validation.metrics import (
    compute_metrics,
    drawdown_series,
    max_drawdown_duration,
    sharpe_ratio,
)


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_constant_positive_return():
    m = compute_metrics(_series([0.01] * 252))
    assert m.volatility == 0.0
    assert m.sharpe == float("inf")
    assert abs(m.cagr - (1.01**252 - 1)) < 1e-9
    assert m.max_drawdown == 0.0


def test_max_drawdown_and_duration():
    r = _series([0.10, -0.10, 0.0, 0.0, 0.12])  # peak 1.1, trough 0.99, recovery 1.1088
    assert abs(drawdown_series(r).min() - (0.99 / 1.1 - 1)) < 1e-12
    assert max_drawdown_duration(r) == 3  # three sessions below the peak
    m = compute_metrics(r)
    assert abs(m.max_drawdown - (0.99 / 1.1 - 1)) < 1e-12
    assert m.max_drawdown_duration_days == 3


def test_sharpe_matches_known_sample():
    rng = np.random.default_rng(0)
    r = _series(rng.normal(0.0004, 0.01, 5000))
    sr = sharpe_ratio(r)
    assert abs(sr - 0.04 * np.sqrt(252)) < 0.3


def test_turnover_and_cost_annualised():
    r = _series([0.0] * 504)
    turnover = pd.Series(0.0, index=r.index)
    turnover.iloc[[10, 300]] = 1.0
    costs = pd.Series(0.0, index=r.index)
    costs.iloc[[10, 300]] = 0.001
    m = compute_metrics(r, turnover=turnover, costs=costs)
    assert abs(m.annual_turnover - 1.0) < 1e-12
    assert abs(m.annual_cost_bps - 10.0) < 1e-9


def test_too_short():
    with pytest.raises(ValueError):
        compute_metrics(_series([0.01]))


def test_as_dict_keys():
    m = compute_metrics(_series([0.01, -0.02, 0.005, 0.0]))
    assert {"cagr", "sharpe", "max_drawdown", "kurtosis"} <= set(m.as_dict())
