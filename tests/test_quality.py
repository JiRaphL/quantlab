import numpy as np
import pandas as pd

from quantlab.data.panel import make_synthetic_panel
from quantlab.data.quality import check_panel


def test_clean_panel_has_no_anomalies(synthetic_panel):
    assert check_panel(synthetic_panel) == []


def test_gap_detected():
    p = make_synthetic_panel(["A"], periods=60)
    # remove 10 consecutive business days in the middle -> a gap > 5 days
    keep = np.r_[0:20, 30:60]
    fields = {
        f: getattr(p, f).iloc[keep] for f in ("close", "open", "high", "low", "volume", "close_raw")
    }
    gapped = type(p)(**fields)
    anomalies = check_panel(gapped, max_gap_days=5)
    assert any(a.kind == "gap" for a in anomalies)


def test_extreme_return_detected():
    p = make_synthetic_panel(["A"], periods=30)
    close = p.close.copy()
    close.iloc[10:, 0] *= 1.8  # +80 % jump on day 10
    bad = p.with_close(close)
    anomalies = check_panel(bad, max_abs_return=0.5)
    assert [a.kind for a in anomalies] == ["extreme_return"]
    assert anomalies[0].date == p.index[10]


def test_non_positive_price_detected():
    p = make_synthetic_panel(["A"], periods=10)
    close = p.close.copy()
    close.iloc[3, 0] = 0.0
    anomalies = check_panel(p.with_close(close))
    assert any(a.kind == "non_positive_price" for a in anomalies)


def test_leading_nan_is_not_an_anomaly():
    p = make_synthetic_panel(["A", "B"], periods=30)
    close = p.close.copy()
    close.iloc[:10, 1] = np.nan  # B starts trading on day 10
    assert check_panel(p.with_close(close)) == []


def test_internal_nan_is_a_gap():
    p = make_synthetic_panel(["A"], periods=30)
    close = p.close.copy()
    close.iloc[10:20, 0] = np.nan
    anomalies = check_panel(p.with_close(close), max_gap_days=5)
    assert any(a.kind == "gap" for a in anomalies)
    assert isinstance(anomalies[0].date, pd.Timestamp)
