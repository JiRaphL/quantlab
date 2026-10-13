"""Look-ahead guard.

Perturb every price strictly after date ``t`` and recompute the strategy. If any weight
up to ``t`` changes, the strategy is peeking into the future. This is run on every
registered strategy in the test suite and can be called on any new idea.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.panel import PricePanel
from quantlab.strategies.base import Strategy


def assert_causal(strategy: Strategy, panel: PricePanel, t: pd.Timestamp, seed: int = 0) -> None:
    base = strategy.target_weights(panel).loc[:t]
    perturbed = perturb_after(panel, t, seed=seed)
    again = strategy.target_weights(perturbed).loc[:t]
    if not np.allclose(base.to_numpy(), again.to_numpy(), atol=1e-12, equal_nan=True):
        diff = (base - again).abs().max().max()
        raise AssertionError(
            f"{strategy!r} is not causal: weights up to {t.date()} changed by up to {diff:.3g} "
            "when prices after t were perturbed"
        )


def perturb_after(panel: PricePanel, t: pd.Timestamp, seed: int = 0) -> PricePanel:
    rng = np.random.default_rng(seed)
    after = panel.index > t
    fields = {}
    for name in ("close", "open", "high", "low", "close_raw"):
        frame = getattr(panel, name).copy()
        noise = rng.uniform(0.5, 1.5, size=(int(after.sum()), frame.shape[1]))
        frame.loc[after] = frame.loc[after].to_numpy() * noise
        fields[name] = frame
    return panel.with_fields(**fields)
