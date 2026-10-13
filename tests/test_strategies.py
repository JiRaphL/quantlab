import numpy as np
import pandas as pd
import pytest

from quantlab.data.panel import PricePanel, make_synthetic_panel
from quantlab.strategies.dual_momentum import DualMomentum
from quantlab.strategies.equal_weight import EqualWeight
from quantlab.strategies.registry import REGISTRY, get_strategy
from quantlab.validation.causality import assert_causal


def _panel_from_close(close: pd.DataFrame) -> PricePanel:
    return PricePanel(
        close=close,
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        volume=close * 0 + 1e6,
        close_raw=close.copy(),
    )


def _trend_panel(periods=60):
    """A rises 1 %/day, B falls 1 %/day, C flat, SAFE rises 0.1 %/day."""
    idx = pd.bdate_range("2020-01-01", periods=periods)
    k = np.arange(periods)
    close = pd.DataFrame(
        {
            "A": 100 * 1.01**k,
            "B": 100 * 0.99**k,
            "C": np.full(periods, 100.0),
            "SAFE": 100 * 1.001**k,
        },
        index=idx,
    )
    return _panel_from_close(close)


def test_equal_weight_is_one_over_n(synthetic_panel):
    w = EqualWeight().target_weights(synthetic_panel)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w.iloc[-1], 0.25)


def test_unknown_param_rejected():
    with pytest.raises(TypeError):
        DualMomentum(lookbak=10)


def test_dual_momentum_picks_best_trending_asset():
    panel = _trend_panel()
    s = DualMomentum(lookback=20, skip=0, top_n=1, absolute_filter=False)
    w = s.target_weights(panel)
    last = w.iloc[-1]
    assert last["A"] == 1.0 and last[["B", "C", "SAFE"]].sum() == 0.0
    assert w.iloc[:20].sum(axis=1).eq(0).all()  # no history yet -> cash


def test_dual_momentum_top2_equal_weight():
    panel = _trend_panel()
    w = DualMomentum(lookback=20, skip=0, top_n=2, absolute_filter=False).target_weights(panel)
    last = w.iloc[-1]
    assert last["A"] == 0.5 and last["SAFE"] == 0.5


def test_absolute_filter_goes_to_cash_when_all_negative():
    idx = pd.bdate_range("2020-01-01", periods=60)
    k = np.arange(60)
    close = pd.DataFrame({"A": 100 * 0.99**k, "B": 100 * 0.98**k}, index=idx)
    panel = _panel_from_close(close)
    w = DualMomentum(lookback=20, skip=0, top_n=2, absolute_filter=True).target_weights(panel)
    assert w.iloc[-1].sum() == 0.0


def test_safe_asset_receives_remainder():
    panel = _trend_panel()
    s = DualMomentum(lookback=20, skip=0, top_n=2, absolute_filter=True, safe_asset="SAFE")
    w = s.target_weights(panel)
    last = w.iloc[-1]
    # A beats SAFE's momentum; B and C do not (B negative, C 0 < SAFE). Only A survives
    # among the top 2 => A gets 1/2, SAFE gets the other half.
    assert last["A"] == 0.5 and last["SAFE"] == 0.5 and last["B"] == 0.0 and last["C"] == 0.0
    assert np.allclose(w.sum(axis=1), 1.0)


def test_safe_asset_must_exist():
    with pytest.raises(KeyError):
        DualMomentum(safe_asset="NOPE").target_weights(_trend_panel())


def test_registry():
    assert get_strategy("dual_momentum") is DualMomentum
    with pytest.raises(KeyError):
        get_strategy("nope")


@pytest.mark.parametrize("name", sorted(REGISTRY))
@pytest.mark.parametrize("pos", [400, 900, 1400])
def test_registered_strategies_are_causal(synthetic_panel, name, pos):
    cls = REGISTRY[name]
    strategy = cls(safe_asset="DDD") if name == "dual_momentum" else cls()
    assert_causal(strategy, synthetic_panel, synthetic_panel.index[pos])


def test_causality_guard_catches_lookahead():
    class Cheater(DualMomentum):
        name = "cheater"

        def target_weights(self, panel):
            w = super().target_weights(panel)
            return w.shift(-5).fillna(0.0)  # uses tomorrow's decision today

    panel = make_synthetic_panel(["A", "B"], periods=600, seed=1)
    with pytest.raises(AssertionError):
        assert_causal(Cheater(lookback=60, skip=0, top_n=1), panel, panel.index[300])
