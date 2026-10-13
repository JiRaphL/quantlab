import numpy as np
import pandas as pd

from quantlab.features.momentum import momentum
from quantlab.features.returns import log_returns, simple_returns
from quantlab.features.trend import above_sma, sma
from quantlab.features.volatility import realized_volatility


def _close():
    idx = pd.bdate_range("2020-01-01", periods=10)
    return pd.DataFrame({"A": np.arange(1, 11, dtype=float)}, index=idx)


def test_simple_and_log_returns():
    c = _close()
    r = simple_returns(c)
    assert np.isnan(r.iloc[0, 0])
    assert r.iloc[1, 0] == 1.0  # 1 -> 2
    lr = log_returns(c)
    assert abs(lr.iloc[1, 0] - np.log(2)) < 1e-12


def test_momentum_definition():
    c = _close()
    m = momentum(c, lookback=3, skip=1)
    # row 3: close[2] / close[0] - 1 = 3/1 - 1 = 2
    assert m.iloc[3, 0] == 2.0
    assert m.iloc[:3, 0].isna().all()


def test_momentum_no_skip():
    c = _close()
    m = momentum(c, lookback=2, skip=0)
    assert m.iloc[2, 0] == 3.0 / 1.0 - 1


def test_realized_volatility_constant_returns_is_zero():
    idx = pd.bdate_range("2020-01-01", periods=100)
    r = pd.DataFrame({"A": 0.01}, index=idx)
    v = realized_volatility(r, window=20)
    assert v.iloc[-1, 0] == 0.0
    assert v.iloc[:19, 0].isna().all()


def test_realized_volatility_annualized():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, 300)}, index=idx)
    v = realized_volatility(r, window=252, annualize=True)
    assert abs(v.iloc[-1, 0] - 0.01 * np.sqrt(252)) < 0.03


def test_sma_and_above_sma():
    c = _close()
    s = sma(c, 3)
    assert s.iloc[2, 0] == 2.0
    a = above_sma(c, 3)
    assert bool(a.iloc[2, 0]) is True  # 3 > 2
    assert bool(a.iloc[0, 0]) is False  # NaN sma -> False
