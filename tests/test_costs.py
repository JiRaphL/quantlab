import pandas as pd

from quantlab.backtest.costs import ZERO_COST, CostModel


def test_turnover_cost_rate():
    cm = CostModel(commission_per_share=0.005, min_commission=1.0, slippage_bps=5.0)
    prices = pd.Series({"A": 100.0, "B": 10.0})
    rate = cm.turnover_cost_rate(prices)
    assert abs(rate["A"] - (0.0005 + 0.00005)) < 1e-15
    assert abs(rate["B"] - (0.0005 + 0.0005)) < 1e-15


def test_turnover_cost_rate_nan_price_is_slippage_only():
    cm = CostModel(commission_per_share=0.005, slippage_bps=5.0)
    rate = cm.turnover_cost_rate(pd.Series({"A": float("nan")}))
    assert abs(rate["A"] - 0.0005) < 1e-15


def test_order_commission():
    cm = CostModel(commission_per_share=0.005, min_commission=1.0)
    assert cm.order_commission(50) == 1.0
    assert cm.order_commission(300) == 1.5
    assert cm.order_commission(-300) == 1.5
    assert cm.order_commission(0) == 0.0


def test_zero_cost():
    assert ZERO_COST.turnover_cost_rate(pd.Series({"A": 50.0}))["A"] == 0.0
    assert ZERO_COST.order_commission(10) == 0.0
