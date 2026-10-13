import numpy as np

from quantlab.backtest.costs import ZERO_COST, CostModel
from quantlab.backtest.engine import run_backtest
from quantlab.portfolio.real_capital import simulate_real_capital
from quantlab.strategies.equal_weight import EqualWeight
from quantlab.validation.metrics import compute_metrics


def test_real_capital_tracks_fractional_when_capital_is_large(synthetic_panel):
    tw = EqualWeight().target_weights(synthetic_panel)
    frac = run_backtest(synthetic_panel, tw, ZERO_COST, initial_capital=1e9)
    real = simulate_real_capital(synthetic_panel, tw, capital=1e9, cost_model=ZERO_COST)
    # With a billion, rounding to whole shares is negligible.
    np.testing.assert_allclose(real.nav.to_numpy(), frac.nav.to_numpy(), rtol=1e-6)
    assert real.config["mode"] == "real_capital"


def test_small_account_pays_more_and_never_goes_negative(synthetic_panel):
    tw = EqualWeight().target_weights(synthetic_panel)
    cm = CostModel(commission_per_share=0.005, min_commission=1.0, slippage_bps=5.0)
    small = simulate_real_capital(synthetic_panel, tw, capital=3_000.0, cost_model=cm)
    big = simulate_real_capital(synthetic_panel, tw, capital=300_000.0, cost_model=cm)
    m_small = compute_metrics(small.returns, turnover=small.turnover, costs=small.costs)
    m_big = compute_metrics(big.returns, turnover=big.turnover, costs=big.costs)
    assert m_small.annual_cost_bps > m_big.annual_cost_bps
    assert (small.cash >= -1e-9).all()
    assert (small.nav > 0).all()
    assert len(small.trades) > 0


def test_weights_are_whole_lots(synthetic_panel):
    tw = EqualWeight().target_weights(synthetic_panel)
    real = simulate_real_capital(synthetic_panel, tw, capital=5_000.0, cost_model=ZERO_COST)
    # 4 symbols at ~100$ => around 12 shares each: actual weights are never exactly 0.25
    last = real.weights.iloc[-1]
    assert (last > 0).all() and not np.allclose(last, 0.25, atol=1e-6)
