from datetime import date

from quantlab.config import BacktestConfig, CostConfig


def _cfg(**over):
    base = dict(
        strategy="dual_momentum",
        params={"lookback": 252, "top_n": 3},
        universe="etf_core",
        start=date(2008, 1, 1),
        end=date(2024, 12, 31),
    )
    base.update(over)
    return BacktestConfig(**base)


def test_hash_is_stable_and_sensitive():
    a, b = _cfg(), _cfg()
    assert a.hash() == b.hash() and len(a.hash()) == 12
    assert _cfg(params={"lookback": 126, "top_n": 3}).hash() != a.hash()
    assert _cfg(cost=CostConfig(slippage_bps=10)).hash() != a.hash()


def test_param_order_does_not_matter():
    a = _cfg(params={"lookback": 252, "top_n": 3})
    b = _cfg(params={"top_n": 3, "lookback": 252})
    assert a.hash() == b.hash()


def test_cost_model_roundtrip():
    cm = _cfg(cost=CostConfig(slippage_bps=7.5)).cost_model()
    assert cm.slippage_bps == 7.5 and cm.min_commission == 1.0


def test_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "strategy: equal_weight\nuniverse: etf_core\nstart: 2010-01-01\nend: 2020-12-31\n"
        "rebalance: weekly\ncost: {slippage_bps: 3}\n"
    )
    cfg = BacktestConfig.from_yaml(p)
    assert cfg.rebalance == "weekly" and cfg.cost.slippage_bps == 3 and cfg.params == {}
