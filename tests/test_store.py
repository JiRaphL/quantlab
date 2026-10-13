from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.data.panel import make_synthetic_panel
from quantlab.data.providers.base import OHLCV_COLUMNS
from quantlab.data.quality import DataQualityError
from quantlab.data.store import PriceStore


def _frame(symbol: str, start: str, periods: int, seed: int = 0) -> pd.DataFrame:
    p = make_synthetic_panel([symbol], start=start, periods=periods, seed=seed)
    df = pd.DataFrame(
        {
            "open": p.open[symbol],
            "high": p.high[symbol],
            "low": p.low[symbol],
            "close": p.close[symbol],
            "volume": p.volume[symbol],
            "close_raw": p.close_raw[symbol],
        }
    )
    df.index.name = "date"
    return df


class FakeProvider:
    name = "fake"

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.calls: list[tuple[str, date, date]] = []

    def fetch(self, symbols, start, end):
        out = {}
        for s in symbols:
            self.calls.append((s, start, end))
            df = self.frames[s]
            out[s] = df.loc[str(start) : str(end)]
        return out


def test_save_and_load_roundtrip(tmp_path):
    store = PriceStore(tmp_path)
    df = _frame("AAA", "2020-01-01", 50)
    store.save("fake", "AAA", df)
    back = store.load("fake", "AAA")
    pd.testing.assert_frame_equal(back, df, check_freq=False)
    assert list(back.columns) == OHLCV_COLUMNS


def test_load_missing_returns_none(tmp_path):
    assert PriceStore(tmp_path).load("fake", "ZZZ") is None


def test_save_merges_and_keeps_old_rows(tmp_path):
    store = PriceStore(tmp_path)
    full = _frame("AAA", "2020-01-01", 60)
    store.save("fake", "AAA", full.iloc[:40])
    store.save("fake", "AAA", full.iloc[30:])  # overlap on 10 rows
    back = store.load("fake", "AAA")
    assert len(back) == 60
    pd.testing.assert_frame_equal(back, full, check_freq=False)


def test_metadata_tracks_saved_symbols(tmp_path):
    store = PriceStore(tmp_path)
    store.save("fake", "AAA", _frame("AAA", "2020-01-01", 10))
    store.save("fake", "BBB", _frame("BBB", "2020-01-01", 20))
    meta = store.metadata()
    assert set(meta["symbol"]) == {"AAA", "BBB"}
    row = meta.set_index("symbol").loc["BBB"]
    assert row["rows"] == 20 and row["provider"] == "fake"


def test_update_fetches_only_missing_range(tmp_path):
    full = _frame("AAA", "2020-01-01", 60)
    provider = FakeProvider({"AAA": full})
    store = PriceStore(tmp_path)
    store.save("fake", "AAA", full.iloc[:30])
    last_saved = full.index[29].date()
    store.update(provider, ["AAA"], end=full.index[-1].date())
    assert provider.calls[0][1] > last_saved - pd.Timedelta(days=10).to_pytimedelta()
    assert len(store.load("fake", "AAA")) == 60


def test_panel_aligns_symbols_with_different_starts(tmp_path):
    store = PriceStore(tmp_path)
    a = _frame("AAA", "2020-01-01", 60, seed=1)
    b = _frame("BBB", "2020-02-03", 37, seed=2)  # starts later
    store.save("fake", "AAA", a)
    store.save("fake", "BBB", b)
    panel = store.panel(["AAA", "BBB"], provider="fake")
    assert panel.symbols == ["AAA", "BBB"]
    assert len(panel.index) == 60
    assert np.isnan(panel.close.iloc[0]["BBB"])
    assert not np.isnan(panel.close.iloc[-1]["BBB"])
    assert panel.close.loc["2020-02-03", "BBB"] == b.loc["2020-02-03", "close"]


def test_panel_date_range(tmp_path):
    store = PriceStore(tmp_path)
    store.save("fake", "AAA", _frame("AAA", "2020-01-01", 60))
    panel = store.panel(["AAA"], start=date(2020, 2, 1), end=date(2020, 2, 29), provider="fake")
    assert panel.index[0] >= pd.Timestamp("2020-02-01")
    assert panel.index[-1] <= pd.Timestamp("2020-02-29")


def test_panel_strict_raises_on_anomaly(tmp_path):
    store = PriceStore(tmp_path)
    df = _frame("AAA", "2020-01-01", 30)
    df.loc[df.index[10:], "close"] *= 1.8
    store.save("fake", "AAA", df)
    with pytest.raises(DataQualityError):
        store.panel(["AAA"], provider="fake")
    panel = store.panel(["AAA"], provider="fake", strict=False)
    assert len(panel.index) == 30


def test_panel_missing_symbol_raises(tmp_path):
    store = PriceStore(tmp_path)
    with pytest.raises(KeyError):
        store.panel(["NOPE"], provider="fake")
