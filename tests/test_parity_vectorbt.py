"""Parity of the home-grown engine against vectorbt, used purely as an oracle.

vectorbt executes ``from_orders`` at the close of the order date with
``size_type="targetpercent"`` relative to the group's value at that close; this is
exactly our convention once targets are shifted to execution dates.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.benchmarks import buy_and_hold_weights, equal_weight_weights
from quantlab.backtest.costs import ZERO_COST, CostModel
from quantlab.backtest.engine import run_backtest
from quantlab.backtest.schedule import rebalance_mask

vbt = pytest.importorskip("vectorbt")

INIT = 100_000.0


def _exec_targets(panel, target_weights, rebalance, lag=1):
    mask = rebalance_mask(panel.index, rebalance)
    return target_weights.where(mask, other=np.nan).shift(lag)


def _vbt_value(panel, exec_targets, slippage=0.0):
    pf = vbt.Portfolio.from_orders(
        close=panel.close,
        size=exec_targets,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
        init_cash=INIT,
        freq="D",
        slippage=slippage,
    )
    return pf.value()


def _random_targets(panel, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.random((len(panel.index), len(panel.symbols)))
    raw = raw / raw.sum(axis=1, keepdims=True) * rng.uniform(0.6, 1.0, size=(len(panel.index), 1))
    return pd.DataFrame(raw, index=panel.index, columns=panel.symbols)


@pytest.mark.parametrize(
    "case, rebalance",
    [
        ("buy_and_hold", "monthly"),
        ("equal_weight", "monthly"),
        ("random", "monthly"),
        ("random", "weekly"),
    ],
)
def test_parity_no_cost(synthetic_panel, case, rebalance):
    panel = synthetic_panel
    if case == "buy_and_hold":
        tw = buy_and_hold_weights(panel, "AAA")
    elif case == "equal_weight":
        tw = equal_weight_weights(panel)
    else:
        tw = _random_targets(panel)
    ours = run_backtest(panel, tw, ZERO_COST, rebalance=rebalance, initial_capital=INIT)
    ref = _vbt_value(panel, _exec_targets(panel, tw, rebalance))
    np.testing.assert_allclose(ours.nav.to_numpy(), ref.to_numpy(), rtol=1e-8)


def test_parity_with_slippage(synthetic_panel):
    """vectorbt applies slippage to the fill price, we charge it on the traded notional.
    The two differ at second order (the size bought at a worse price is slightly smaller),
    hence a looser tolerance than the exact no-cost case."""
    panel = synthetic_panel
    tw = _random_targets(panel, seed=3)
    cm = CostModel(commission_per_share=0.0, min_commission=0.0, slippage_bps=10.0)
    ours = run_backtest(panel, tw, cm, rebalance="monthly", initial_capital=INIT)
    ref = _vbt_value(panel, _exec_targets(panel, tw, "monthly"), slippage=0.001)
    np.testing.assert_allclose(ours.nav.to_numpy(), ref.to_numpy(), rtol=2e-4)
    # And costs do bite in both: final NAV below the no-cost run.
    free = run_backtest(panel, tw, ZERO_COST, rebalance="monthly", initial_capital=INIT)
    assert ours.nav.iloc[-1] < free.nav.iloc[-1]
