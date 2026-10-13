from datetime import date

import pandas as pd
import pytest

from quantlab.config import BacktestConfig
from quantlab.research.registry import ExperimentRegistry
from quantlab.validation.metrics import compute_metrics

RATIONALE = (
    "Trailing 12-month returns persist for a few months because of investor under-reaction "
    "and slow capital flows (Jegadeesh & Titman 1993)."
)


def _metrics(seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    r = pd.Series(rng.normal(0.0004, 0.01, 500), index=pd.bdate_range("2020-01-01", periods=500))
    return compute_metrics(r)


def _cfg(**over):
    base = dict(
        strategy="dual_momentum",
        params={"lookback": 252},
        universe="etf_core",
        start=date(2010, 1, 1),
        end=date(2020, 1, 1),
    )
    base.update(over)
    return BacktestConfig(**base)


def test_short_rationale_rejected(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.duckdb")
    with pytest.raises(ValueError):
        reg.register_hypothesis("H1", "momentum works")


def test_register_and_log_runs(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.duckdb")
    h = reg.register_hypothesis("H1 momentum", RATIONALE)
    assert h == 1
    assert reg.hypothesis(h)["name"] == "H1 momentum"
    r1 = reg.log_run(h, _cfg(), _metrics(0), report_path="r1.html")
    r2 = reg.log_run(h, _cfg(params={"lookback": 126}), _metrics(1), n_trials_declared=18)
    assert (r1, r2) == (1, 2)
    assert reg.n_trials(h) == 19
    runs = reg.runs(h)
    assert len(runs) == 2
    assert {"sharpe", "max_drawdown", "param_lookback", "config_hash"} <= set(runs.columns)
    assert runs.iloc[1]["param_lookback"] == 126


def test_unknown_hypothesis_rejected(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.duckdb")
    with pytest.raises(KeyError):
        reg.log_run(42, _cfg(), _metrics())


def test_registry_persists_across_instances(tmp_path):
    path = tmp_path / "exp.duckdb"
    h = ExperimentRegistry(path).register_hypothesis("H1", RATIONALE)
    again = ExperimentRegistry(path)
    assert len(again.hypotheses()) == 1 and again.n_trials(h) == 0
    assert again.runs().empty
