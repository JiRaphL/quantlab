import pandas as pd

from quantlab.backtest.costs import ZERO_COST, CostModel
from quantlab.portfolio.lots import discretize_weights


def test_whole_shares_and_cash():
    target = pd.Series({"A": 1.0})
    prices = pd.Series({"A": 500.0})
    alloc = discretize_weights(target, prices, capital=1000.0, cost_model=ZERO_COST)
    assert alloc.shares["A"] == 2 and alloc.cash == 0.0 and alloc.weights["A"] == 1.0


def test_rounding_down_leaves_cash():
    alloc = discretize_weights(pd.Series({"A": 1.0}), pd.Series({"A": 300.0}), 1000.0, ZERO_COST)
    assert alloc.shares["A"] == 3 and alloc.cash == 100.0


def test_min_commission_reduces_cash():
    cm = CostModel(commission_per_share=0.005, min_commission=1.0, slippage_bps=0.0)
    alloc = discretize_weights(
        pd.Series({"A": 0.5, "B": 0.5}), pd.Series({"A": 10.0, "B": 10.0}), 1000.0, cm
    )
    assert alloc.shares.tolist() == [50, 50]
    assert alloc.commissions == 2.0  # two orders at the 1 $ minimum
    assert alloc.cash == 1000.0 - 1000.0 - 2.0  # negative by the commissions: caller sizes down


def test_orders_relative_to_current_holdings():
    cm = CostModel(commission_per_share=0.005, min_commission=1.0, slippage_bps=0.0)
    current = pd.Series({"A": 10, "B": 0})
    alloc = discretize_weights(
        pd.Series({"A": 0.0, "B": 1.0}),
        pd.Series({"A": 10.0, "B": 20.0}),
        1000.0,
        cm,
        current_shares=current,
    )
    assert alloc.orders["A"] == -10 and alloc.orders["B"] == 50
    assert alloc.commissions == 2.0


def test_untradable_symbol_keeps_current_shares():
    current = pd.Series({"A": 5, "B": 0})
    alloc = discretize_weights(
        pd.Series({"A": 0.5, "B": 0.5}),
        pd.Series({"A": float("nan"), "B": 10.0}),
        1000.0,
        ZERO_COST,
        current_shares=current,
    )
    assert alloc.shares["A"] == 5 and alloc.orders["A"] == 0 and alloc.shares["B"] == 50
