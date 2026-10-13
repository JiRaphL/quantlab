from datetime import date
from pathlib import Path

import pytest

from quantlab.data.universe import load_universe, load_universe_file


def test_etf_core_loads():
    u = load_universe("etf_core")
    assert u.name == "etf_core"
    assert u.survivorship_bias is False
    assert len(u.symbols) == 14
    assert "SPY" in u.tickers and "IEF" in u.tickers


def test_active_on_respects_inception():
    u = load_universe("etf_core")
    active = u.active_on(date(2000, 1, 1))
    assert "SPY" in active and "QQQ" in active
    assert "IWM" not in active  # inception 2000-05-26


def test_missing_universe_raises():
    with pytest.raises(FileNotFoundError):
        load_universe("does_not_exist")


def test_duplicate_tickers_rejected(tmp_path: Path):
    yaml_text = """
name: dup
description: test
survivorship_bias: false
symbols:
  - {symbol: SPY, inception: 1993-01-29, description: a, asset_class: equity}
  - {symbol: SPY, inception: 1993-01-29, description: b, asset_class: equity}
"""
    path = tmp_path / "dup.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError):
        load_universe_file(path)
