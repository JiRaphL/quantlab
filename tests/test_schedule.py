import pandas as pd

from quantlab.backtest.schedule import rebalance_mask


def test_monthly_mask_last_business_day():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    m = rebalance_mask(idx, "monthly")
    assert m.sum() == 3
    assert list(m[m].index) == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
        pd.Timestamp("2020-03-31"),
    ]


def test_monthly_mask_uses_available_dates_not_calendar():
    # Remove 2020-01-31 from the index: January's rebalance must fall on the 30th.
    idx = pd.bdate_range("2020-01-01", "2020-02-29").drop(pd.Timestamp("2020-01-31"))
    m = rebalance_mask(idx, "monthly")
    assert m[pd.Timestamp("2020-01-30")]


def test_weekly_mask():
    idx = pd.bdate_range("2020-01-06", "2020-01-17")  # two full weeks
    m = rebalance_mask(idx, "weekly")
    assert list(m[m].index) == [pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-17")]


def test_daily_mask_all_true():
    idx = pd.bdate_range("2020-01-01", periods=5)
    assert rebalance_mask(idx, "daily").all()


def test_unknown_freq():
    import pytest

    with pytest.raises(ValueError):
        rebalance_mask(pd.bdate_range("2020-01-01", periods=5), "yearly")  # type: ignore[arg-type]
