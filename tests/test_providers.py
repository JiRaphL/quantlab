from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.data.providers.base import OHLCV_COLUMNS, normalize_ohlcv


def _raw_frame():
    idx = pd.DatetimeIndex(
        ["2020-01-02", "2020-01-03", "2020-01-03", "2020-01-06"], tz="America/New_York"
    )
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0, 102.0, 104.0],
            "High": [101.0, 103.0, 103.0, 105.0],
            "Low": [99.0, 101.0, 101.0, 103.0],
            "Close": [100.0, 102.0, 102.5, 104.0],
            "Adj Close": [50.0, 51.0, 51.25, 52.0],
            "Volume": [1000, 2000, 2500, 3000],
        },
        index=idx,
    )


def test_normalize_columns_and_dtypes():
    out = normalize_ohlcv(_raw_frame())
    assert list(out.columns) == OHLCV_COLUMNS
    assert all(out[c].dtype == np.float64 for c in OHLCV_COLUMNS)
    assert out.index.tz is None
    assert out.index.name == "date"


def test_normalize_applies_adjustment_factor():
    out = normalize_ohlcv(_raw_frame())
    first = out.iloc[0]
    assert first["close"] == 50.0
    assert first["close_raw"] == 100.0
    assert first["open"] == 50.0  # factor 0.5 applied to open
    assert first["high"] == 50.5 and first["low"] == 49.5


def test_normalize_drops_duplicate_dates_keep_last():
    out = normalize_ohlcv(_raw_frame())
    assert len(out) == 3
    assert out.loc["2020-01-03", "close_raw"] == 102.5


def test_normalize_sorted_ascending():
    raw = _raw_frame().iloc[::-1]
    out = normalize_ohlcv(raw)
    assert out.index.is_monotonic_increasing


@pytest.mark.network
def test_yfinance_fetch_spy():
    from quantlab.data.providers.yfinance_provider import YFinanceProvider

    data = YFinanceProvider().fetch(["SPY"], date(2024, 1, 2), date(2024, 1, 16))
    df = data["SPY"]
    assert list(df.columns) == OHLCV_COLUMNS
    assert 8 <= len(df) <= 11
