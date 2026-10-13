"""Tests for the pieces added for H2: rebalance bands, fixed-weight benchmark,
volatility-scaled momentum."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.benchmarks import fixed_weights
from quantlab.backtest.costs import ZERO_COST, CostModel
from quantlab.backtest.engine import run_backtest
from quantlab.config import BacktestConfig
from quantlab.data.panel import PricePanel
from quantlab.portfolio.real_capital import simulate_real_capital
from quantlab.strategies.vol_scaled_momentum import VolScaledMomentum


def _panel_from_close(close: pd.DataFrame) -> PricePanel:
    return PricePanel(
        close=close,
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        volume=close * 0 + 1e6,
        close_raw=close.copy(),
    )


# -- rebalance band -------------------------------------------------------------------


def test_band_skips_small_deviations(synthetic_panel):
    tw = fixed_weights(synthetic_panel, {"AAA": 0.5, "BBB": 0.5})
    cm = CostModel(commission_per_share=0.0, min_commission=0.0, slippage_bps=10.0)
    tight = run_backtest(synthetic_panel, tw, cm, "monthly")
    banded = run_backtest(synthetic_panel, tw, cm, "monthly", rebalance_band=0.05)
    assert (banded.turnover > 0).sum() < (tight.turnover > 0).sum()
    assert banded.costs.sum() < tight.costs.sum()
    # The first trade (from all-cash) always happens: deviation is 0.5.
    assert banded.turnover[banded.turnover > 0].iloc[0] == 1.0


def test_band_zero_is_default_behaviour(synthetic_panel):
    tw = fixed_weights(synthetic_panel, {"AAA": 0.5, "BBB": 0.5})
    a = run_backtest(synthetic_panel, tw, ZERO_COST, "monthly")
    b = run_backtest(synthetic_panel, tw, ZERO_COST, "monthly", rebalance_band=0.0)
    pd.testing.assert_series_equal(a.nav, b.nav)


def test_band_rejects_negative(synthetic_panel):
    tw = fixed_weights(synthetic_panel, {"AAA": 1.0})
    with pytest.raises(ValueError):
        run_backtest(synthetic_panel, tw, ZERO_COST, rebalance_band=-0.1)


def test_band_in_real_capital(synthetic_panel):
    tw = fixed_weights(synthetic_panel, {"AAA": 0.5, "BBB": 0.5})
    cm = CostModel(commission_per_share=0.005, min_commission=1.0, slippage_bps=5.0)
    tight = simulate_real_capital(synthetic_panel, tw, 10_000.0, cm, "monthly")
    banded = simulate_real_capital(
        synthetic_panel, tw, 10_000.0, cm, "monthly", rebalance_band=0.05
    )
    assert len(banded.trades) < len(tight.trades)


# -- fixed-weight benchmark -----------------------------------------------------------


def test_fixed_weights_60_40(synthetic_panel):
    w = fixed_weights(synthetic_panel, {"AAA": 0.6, "BBB": 0.4})
    assert w.iloc[-1].tolist() == [0.6, 0.4, 0.0, 0.0]


def test_fixed_weights_respect_missing_prices():
    idx = pd.bdate_range("2020-01-01", periods=10)
    close = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    close.iloc[:5, 1] = np.nan
    w = fixed_weights(_panel_from_close(close), {"A": 0.6, "B": 0.4})
    assert w.iloc[0]["B"] == 0.0 and w.iloc[6]["B"] == 0.4


def test_fixed_weights_validation(synthetic_panel):
    with pytest.raises(KeyError):
        fixed_weights(synthetic_panel, {"NOPE": 1.0})
    with pytest.raises(ValueError):
        fixed_weights(synthetic_panel, {"AAA": 0.8, "BBB": 0.5})


def test_config_benchmark_name():
    base = dict(strategy="equal_weight", universe="etf_core", start="2010-01-01", end="2020-01-01")
    assert BacktestConfig(**base, benchmark="SPY").benchmark_name() == "SPY"
    cfg = BacktestConfig(**base, benchmark={"SPY": 0.6, "IEF": 0.4})
    assert cfg.benchmark_name() == "60 SPY/40 IEF"
    assert cfg.hash() != BacktestConfig(**base).hash()


# -- volatility-scaled momentum -------------------------------------------------------


def _two_trends(periods=200, seed=0):
    """A: strong trend, high vol. B: weaker trend, very low vol => better risk-adjusted."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=periods)
    a = 100 * np.exp(np.cumsum(rng.normal(0.0020, 0.030, periods)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0010, 0.002, periods)))
    c = 100 * np.exp(np.cumsum(rng.normal(-0.0010, 0.010, periods)))
    return _panel_from_close(pd.DataFrame({"A": a, "B": b, "C": c}, index=idx))


def test_ranks_on_risk_adjusted_momentum():
    panel = _two_trends()
    s = VolScaledMomentum(
        lookback=100, skip=0, vol_window=40, top_n=1, target_vol=1.0, absolute_filter=False
    )
    w = s.target_weights(panel)
    assert w.iloc[-1]["B"] > 0 and w.iloc[-1]["A"] == 0.0


def test_inverse_vol_weights_and_sum():
    panel = _two_trends()
    s = VolScaledMomentum(
        lookback=100, skip=0, vol_window=40, top_n=2, target_vol=1.0, absolute_filter=False
    )
    last = s.target_weights(panel).iloc[-1]
    assert last["B"] > last["A"] > 0 and last["C"] == 0.0
    assert abs(last.sum() - 1.0) < 1e-12


def test_target_vol_scales_down_and_remainder_goes_to_safe():
    panel = _two_trends()
    free = VolScaledMomentum(
        lookback=100,
        skip=0,
        vol_window=40,
        top_n=1,
        target_vol=1.0,
        absolute_filter=False,
        safe_asset="C",
    )
    capped = VolScaledMomentum(
        lookback=100,
        skip=0,
        vol_window=40,
        top_n=1,
        target_vol=0.01,
        absolute_filter=False,
        safe_asset="C",
    )
    wf, wc = free.target_weights(panel).iloc[-1], capped.target_weights(panel).iloc[-1]
    assert wf["B"] == 1.0 and wf["C"] == 0.0
    assert 0 < wc["B"] < 1.0 and abs(wc["B"] + wc["C"] - 1.0) < 1e-12


def test_absolute_filter_uses_raw_momentum():
    panel = _two_trends()
    s = VolScaledMomentum(
        lookback=100, skip=0, vol_window=40, top_n=3, target_vol=1.0, absolute_filter=True
    )
    last = s.target_weights(panel).iloc[-1]
    assert last["C"] == 0.0  # negative trend never held
    assert last.sum() < 1.0  # unfilled slot stays defensive (cash)


def test_weights_are_valid_on_synthetic(synthetic_panel):
    w = VolScaledMomentum(safe_asset="DDD").target_weights(synthetic_panel)
    assert (w >= 0).all().all()
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()
    run_backtest(synthetic_panel, w, ZERO_COST)  # engine accepts it
