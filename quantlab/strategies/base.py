"""Strategy contract.

A strategy maps a :class:`PricePanel` to target weights: one row per date, one column per
symbol, each row being the allocation decided with data up to and including that date.
It knows nothing about capital, positions, costs or execution timing; the engine handles
all of that. Implementations must be causal: only ``rolling``, ``expanding`` and
positive ``shift`` operations on the panel. The causality guard in
:mod:`quantlab.validation.causality` verifies it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from quantlab.data.panel import PricePanel


class Strategy(ABC):
    name: ClassVar[str] = "abstract"

    def __init__(self, **params: Any):
        defaults = self.default_params()
        unknown = set(params) - set(defaults)
        if unknown:
            raise TypeError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params: dict[str, Any] = {**defaults, **params}

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def param_grid_default(cls) -> dict[str, list]:
        """A reasonable grid for sensitivity sweeps and walk-forward selection."""
        return {}

    @abstractmethod
    def target_weights(self, panel: PricePanel) -> pd.DataFrame: ...

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name, "params": dict(self.params)}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"
