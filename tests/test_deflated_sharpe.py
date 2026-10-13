import numpy as np

from quantlab.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def test_psr_at_benchmark_is_half():
    assert abs(probabilistic_sharpe_ratio(0.05, 0.05, 500) - 0.5) < 1e-12


def test_psr_increases_with_sample_size():
    a = probabilistic_sharpe_ratio(0.05, 0.0, 100)
    b = probabilistic_sharpe_ratio(0.05, 0.0, 1000)
    assert 0.5 < a < b < 1.0


def test_psr_penalises_negative_skew_and_fat_tails():
    normal = probabilistic_sharpe_ratio(0.05, 0.0, 500, skew=0.0, kurtosis=3.0)
    ugly = probabilistic_sharpe_ratio(0.05, 0.0, 500, skew=-1.0, kurtosis=8.0)
    assert ugly < normal


def test_expected_max_sharpe():
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(10, 0.0) == 0.0
    e10 = expected_max_sharpe(10, 0.01)
    e100 = expected_max_sharpe(100, 0.01)
    assert 0 < e10 < e100
    # 100 trials with daily-Sharpe sd 0.1 => expected max around 2.5 sd
    assert abs(e100 / 0.1 - 2.5) < 0.3


def test_dsr_of_lucky_zero_strategy_is_low():
    # 100 trials with sd of daily Sharpe 0.03; the best came out at 0.06 (2 sd): not
    # impressive given the search.
    dsr = deflated_sharpe_ratio(sr=0.06, n_trials=100, sr_variance=0.03**2, n=1500)
    assert dsr < 0.5
    # Same Sharpe with a single trial is convincing.
    assert deflated_sharpe_ratio(sr=0.06, n_trials=1, sr_variance=0.03**2, n=1500) > 0.95


def test_dsr_nan_when_denominator_invalid():
    assert np.isnan(probabilistic_sharpe_ratio(10.0, 0.0, 100, skew=5.0, kurtosis=1.0))
