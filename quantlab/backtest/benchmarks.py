"""Reference allocations every strategy is compared against."""

from __future__ import annotations

import pandas as pd

from quantlab.data.panel import PricePanel


def buy_and_hold_weights(panel: PricePanel, symbol: str) -> pd.DataFrame:
    """100 % in ``symbol`` whenever it has a price, else cash."""
    if symbol not in panel.symbols:
        raise KeyError(symbol)
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    w[symbol] = panel.close[symbol].notna().astype(float)
    return w


def equal_weight_weights(panel: PricePanel) -> pd.DataFrame:
    """1/n over the symbols that have a price at each date."""
    available = panel.close.notna().astype(float)
    n = available.sum(axis=1).replace(0.0, float("nan"))
    return available.div(n, axis=0).fillna(0.0)


def fixed_weights(panel: PricePanel, weights: dict[str, float]) -> pd.DataFrame:
    """Constant target weights (e.g. 60/40), applied whenever each symbol has a price.
    Missing symbols' weight stays in cash. Weights must sum to <= 1."""
    unknown = set(weights) - set(panel.symbols)
    if unknown:
        raise KeyError(f"symbols not in panel: {sorted(unknown)}")
    if sum(weights.values()) > 1.0 + 1e-9 or any(v < 0 for v in weights.values()):
        raise ValueError("fixed weights must be >= 0 and sum to <= 1")
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    for symbol, weight in weights.items():
        w[symbol] = panel.close[symbol].notna().astype(float) * weight
    return w
