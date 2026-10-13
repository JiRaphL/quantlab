import numpy as np
import pandas as pd

from quantlab.validation.bootstrap import block_bootstrap_sharpe
from quantlab.validation.metrics import sharpe_ratio


def _returns(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=pd.bdate_range("2000-01-01", periods=n))


def test_ci_contains_point_estimate():
    r = _returns(1500)
    lo, hi, samples = block_bootstrap_sharpe(r, n_boot=300)
    assert lo < sharpe_ratio(r) < hi
    assert len(samples) == 300


def test_ci_narrows_with_more_data():
    lo1, hi1, _ = block_bootstrap_sharpe(_returns(500), n_boot=300)
    lo2, hi2, _ = block_bootstrap_sharpe(_returns(5000), n_boot=300)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_reproducible():
    r = _returns(800)
    a = block_bootstrap_sharpe(r, n_boot=50, seed=1)
    b = block_bootstrap_sharpe(r, n_boot=50, seed=1)
    assert a[0] == b[0] and a[1] == b[1]
