"""Name -> strategy class. Add new strategies here so the CLI, the walk-forward and the
causality tests see them."""

from __future__ import annotations

from quantlab.strategies.base import Strategy
from quantlab.strategies.dual_momentum import DualMomentum
from quantlab.strategies.equal_weight import EqualWeight
from quantlab.strategies.vol_scaled_momentum import VolScaledMomentum

REGISTRY: dict[str, type[Strategy]] = {
    EqualWeight.name: EqualWeight,
    DualMomentum.name: DualMomentum,
    VolScaledMomentum.name: VolScaledMomentum,
}


def get_strategy(name: str) -> type[Strategy]:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(REGISTRY)}") from exc
