import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.benchmarks import buy_and_hold_weights, equal_weight_weights
from quantlab.backtest.costs import ZERO_COST, CostModel
from quantlab.backtest.engine import run_backtest
from quantlab.data.panel import PricePanel


def _panel_from_close(close: pd.DataFrame) -> PricePanel:
    return PricePanel(
        close=close,
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        volume=close * 0 + 1e6,
        close_raw=close.copy(),
    )


def _growth_panel(rate=0.01, periods=30, symbols=("A",)):
    idx = pd.bdate_range("2020-01-01", periods=periods)
    close = pd.DataFrame({s: 100.0 * (1 + rate) ** np.arange(periods) for s in symbols}, index=idx)
    return _panel_from_close(close)


def test_constant_return_single_asset_no_cost():
    panel = _growth_panel(0.01, 30)
    tw = buy_and_hold_weights(panel, "A")
    res = run_backtest(panel, tw, ZERO_COST, rebalance="daily", initial_capital=1000.0)
    # lag 1: nothing held on day 0, bought at close of day 1, so NAV compounds from day 2.
    assert res.nav.iloc[0] == 1000.0 and res.nav.iloc[1] == 1000.0
    expected = 1000.0 * 1.01 ** (np.arange(30) - 1)
    np.testing.assert_allclose(res.nav.to_numpy()[2:], expected[2:], rtol=1e-12)
    assert res.returns.iloc[0] == 0.0 and abs(res.returns.iloc[5] - 0.01) < 1e-12


def test_lag_applied_to_weights():
    panel = _growth_panel(0.0, 10)
    tw = buy_and_hold_weights(panel, "A")
    res = run_backtest(panel, tw, ZERO_COST, rebalance="daily")
    assert res.weights.iloc[0, 0] == 0.0
    assert res.weights.iloc[1, 0] == 1.0
    res0 = run_backtest(panel, tw, ZERO_COST, rebalance="daily", execution_lag=0)
    assert res0.weights.iloc[0, 0] == 1.0


def test_fifty_fifty_drift_and_turnover():
    # A flat, B +10 % on day 2 only. Monthly rebalance => signals on the last day of month.
    idx = pd.bdate_range("2020-01-27", periods=10)  # spans Jan 31 (signal) -> Feb 3 (exec)
    a = np.full(10, 100.0)
    b = np.full(10, 100.0)
    b[6:] = 110.0  # jump between idx[5] and idx[6] (after execution on Feb 3 = idx[5])
    close = pd.DataFrame({"A": a, "B": b}, index=idx)
    panel = _panel_from_close(close)
    tw = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
    res = run_backtest(panel, tw, ZERO_COST, rebalance="monthly")
    exec_date = pd.Timestamp("2020-02-03")
    assert res.weights.loc[exec_date].tolist() == [0.5, 0.5]
    assert res.turnover.loc[exec_date] == 1.0
    # After B's +10 %, weights drift to 0.5/1.05 and 0.55/1.05.
    after = res.weights.loc[idx[6]]
    assert abs(after["A"] - 0.5 / 1.05) < 1e-12 and abs(after["B"] - 0.55 / 1.05) < 1e-12
    assert abs(res.nav.iloc[-1] / res.nav.iloc[0] - 1.05) < 1e-12


def test_costs_reduce_nav_by_turnover_times_rate():
    panel = _growth_panel(0.0, 10)
    tw = buy_and_hold_weights(panel, "A")
    cm = CostModel(commission_per_share=0.0, min_commission=0.0, slippage_bps=10.0)
    res = run_backtest(panel, tw, cm, rebalance="daily", initial_capital=1000.0)
    # One trade of turnover 1.0 at 10 bps => NAV 999, then flat.
    assert abs(res.nav.iloc[1] - 999.0) < 1e-9
    assert abs(res.nav.iloc[-1] - 999.0) < 1e-9
    assert res.nav_gross.iloc[-1] == 1000.0
    assert res.costs.iloc[1] == 0.001
    assert len(res.trades) == 1 and abs(res.trades.iloc[0]["cost"] - 1.0) < 1e-9


def test_commission_per_share_depends_on_raw_price():
    panel = _growth_panel(0.0, 5)
    tw = buy_and_hold_weights(panel, "A")
    cm = CostModel(commission_per_share=0.005, min_commission=0.0, slippage_bps=0.0)
    res = run_backtest(panel, tw, cm, rebalance="daily", initial_capital=1000.0)
    # price 100 => 0.005/100 = 5e-5 of notional
    assert abs(res.costs.iloc[1] - 5e-5) < 1e-15


def test_rejects_negative_or_leveraged_weights():
    panel = _growth_panel(0.0, 5, symbols=("A", "B"))
    neg = pd.DataFrame({"A": -0.1, "B": 0.5}, index=panel.index)
    with pytest.raises(ValueError):
        run_backtest(panel, neg, ZERO_COST)
    lev = pd.DataFrame({"A": 0.8, "B": 0.5}, index=panel.index)
    with pytest.raises(ValueError):
        run_backtest(panel, lev, ZERO_COST)


def test_nan_prices_force_zero_weight():
    panel = _growth_panel(0.0, 10, symbols=("A", "B"))
    close = panel.close.copy()
    close.iloc[:5, 1] = np.nan  # B starts on day 5
    panel = _panel_from_close(close)
    tw = pd.DataFrame(0.5, index=panel.index, columns=["A", "B"])
    res = run_backtest(panel, tw, ZERO_COST, rebalance="daily")
    assert res.weights.iloc[2]["B"] == 0.0 and res.weights.iloc[2]["A"] == 0.5
    assert res.weights.iloc[7]["B"] == 0.5


def test_no_trade_when_target_unchanged():
    panel = _growth_panel(0.0, 10)
    tw = buy_and_hold_weights(panel, "A")
    res = run_backtest(panel, tw, CostModel(slippage_bps=10.0), rebalance="daily")
    assert (res.turnover.iloc[2:] == 0).all()
    assert len(res.trades) == 1


def test_equal_weight_benchmark_sums_to_one(synthetic_panel):
    tw = equal_weight_weights(synthetic_panel)
    assert np.allclose(tw.sum(axis=1), 1.0)
    res = run_backtest(synthetic_panel, tw, ZERO_COST, rebalance="monthly")
    assert np.allclose(res.weights.iloc[-1].sum(), 1.0)
    assert res.config["rebalance"] == "monthly"


def test_result_cash_property(synthetic_panel):
    tw = equal_weight_weights(synthetic_panel) * 0.5
    res = run_backtest(synthetic_panel, tw, ZERO_COST, rebalance="monthly")
    assert abs(res.cash.iloc[-1] - 0.5) < 0.05  # drifts a little between rebalances
