import numpy as np
import pytest

from quantlab.data.panel import PricePanel, make_synthetic_panel


def test_synthetic_panel_shapes():
    p = make_synthetic_panel(["A", "B"], periods=100, seed=1)
    assert p.symbols == ["A", "B"]
    assert len(p.index) == 100
    assert p.close.shape == p.close_raw.shape == (100, 2)
    assert (p.close > 0).all().all()


def test_synthetic_panel_is_reproducible():
    a = make_synthetic_panel(["A"], periods=20, seed=7)
    b = make_synthetic_panel(["A"], periods=20, seed=7)
    assert a.close.equals(b.close)


def test_returns_first_row_nan():
    p = make_synthetic_panel(["A"], periods=10)
    r = p.returns()
    assert np.isnan(r.iloc[0, 0]) and r.shape == (10, 1)
    expected = p.close.iloc[1, 0] / p.close.iloc[0, 0] - 1
    assert abs(r.iloc[1, 0] - expected) < 1e-12


def test_until_and_slice():
    p = make_synthetic_panel(["A"], periods=50)
    d = p.index[9]
    assert len(p.until(d).index) == 10
    assert len(p.slice(start=p.index[10], end=p.index[19]).index) == 10


def test_with_close_replaces_close_only():
    p = make_synthetic_panel(["A"], periods=5)
    q = p.with_close(p.close * 2)
    assert (q.close == p.close * 2).all().all()
    assert q.open.equals(p.open)


def test_mismatched_columns_rejected():
    p = make_synthetic_panel(["A", "B"], periods=5)
    with pytest.raises(ValueError):
        PricePanel(
            close=p.close[["A"]],
            open=p.open,
            high=p.high,
            low=p.low,
            volume=p.volume,
            close_raw=p.close_raw,
        )


def test_unsorted_index_rejected():
    p = make_synthetic_panel(["A"], periods=5)
    rev = p.close.iloc[::-1]
    with pytest.raises(ValueError):
        PricePanel(close=rev, open=rev, high=rev, low=rev, volume=rev, close_raw=rev)
