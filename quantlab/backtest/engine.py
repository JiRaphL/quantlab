"""The backtest engine: target weights in, net-of-cost performance out.

Conventions (the most important rules of the project):

* ``target_weights.loc[t]`` is the allocation decided with information up to the close
  of ``t``. The engine only *reads* targets on rebalance dates (see
  :func:`quantlab.backtest.schedule.rebalance_mask`).
* Trades happen ``execution_lag`` sessions later, at that session's close, after the
  day's return has accrued. With the default lag of 1: signal at close t, trade at
  close t+1. A strategy has no way to shorten this.
* Between rebalances, weights drift with returns. Cash earns nothing.
* Costs are charged on the traded notional at execution:
  ``sum_i |w_target_i - w_held_i| * cost_rate_i``.

The loop is written day by day on numpy arrays on purpose: it is slow enough to read and
fast enough for daily data (a 30-year, 500-symbol run takes well under a second).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.backtest.costs import IBKR_DEFAULT, CostModel
from quantlab.backtest.result import BacktestResult
from quantlab.backtest.schedule import Rebalance, rebalance_mask
from quantlab.data.panel import PricePanel

WEIGHT_TOL = 1e-9


def validate_target_weights(target_weights: pd.DataFrame, panel: PricePanel) -> pd.DataFrame:
    """Align to the panel, replace NaN by 0, reject negative or leveraged rows."""
    tw = target_weights.reindex(index=panel.index, columns=panel.symbols).fillna(0.0)
    if (tw < -WEIGHT_TOL).any().any():
        raise ValueError("target weights must be >= 0 (long only)")
    if (tw.sum(axis=1) > 1.0 + WEIGHT_TOL).any():
        raise ValueError("target weights must sum to <= 1 (no leverage)")
    return tw.clip(lower=0.0)


def run_backtest(
    panel: PricePanel,
    target_weights: pd.DataFrame,
    cost_model: CostModel = IBKR_DEFAULT,
    rebalance: Rebalance = "monthly",
    initial_capital: float = 100_000.0,
    execution_lag: int = 1,
    config: dict | None = None,
    rebalance_band: float = 0.0,
) -> BacktestResult:
    """``rebalance_band``: on an execution date, trade only if some asset's weight deviates
    from its target by more than this amount (absolute weight). 0 = always rebalance.
    Bands cut turnover a lot for strategies whose targets barely move month to month."""
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    if rebalance_band < 0:
        raise ValueError("rebalance_band must be >= 0")

    index = panel.index
    symbols = panel.symbols
    n_dates, n_assets = len(index), len(symbols)

    targets = validate_target_weights(target_weights, panel)
    signal_mask = rebalance_mask(index, rebalance)
    exec_targets = targets.where(signal_mask, other=np.nan).shift(execution_lag)
    is_exec_date = exec_targets.notna().any(axis=1).to_numpy()
    exec_arr = exec_targets.fillna(0.0).to_numpy()

    rets = panel.returns().to_numpy()
    rets = np.where(np.isnan(rets), 0.0, rets)
    tradable = panel.close.notna().to_numpy()
    cost_rates = np.vstack(
        [cost_model.turnover_cost_rate(panel.close_raw.iloc[i]).to_numpy() for i in range(n_dates)]
    )

    w = np.zeros(n_assets)
    nav = float(initial_capital)
    nav_gross = float(initial_capital)

    nav_out = np.empty(n_dates)
    nav_gross_out = np.empty(n_dates)
    weights_out = np.zeros((n_dates, n_assets))
    turnover_out = np.zeros(n_dates)
    costs_out = np.zeros(n_dates)
    trade_rows: list[tuple] = []

    for t in range(n_dates):
        r = rets[t]
        r_p = float(w @ r)
        nav *= 1.0 + r_p
        nav_gross *= 1.0 + r_p
        if 1.0 + r_p != 0.0:
            w = w * (1.0 + r) / (1.0 + r_p)
        else:
            w = np.zeros(n_assets)

        if is_exec_date[t]:
            target = exec_arr[t] * tradable[t]  # cannot hold what has no price
            delta = target - w
            turnover = float(np.abs(delta).sum())
            cost_frac = float((np.abs(delta) * cost_rates[t]).sum())
            if turnover > WEIGHT_TOL and float(np.abs(delta).max()) > rebalance_band:
                nav_before = nav
                nav *= 1.0 - cost_frac
                turnover_out[t] = turnover
                costs_out[t] = cost_frac
                for j in np.where(np.abs(delta) > WEIGHT_TOL)[0]:
                    trade_rows.append(
                        (
                            index[t],
                            symbols[j],
                            float(delta[j]),
                            float(delta[j] * nav_before),
                            float(abs(delta[j]) * cost_rates[t, j] * nav_before),
                        )
                    )
                w = target

        nav_out[t] = nav
        nav_gross_out[t] = nav_gross
        weights_out[t] = w

    nav_s = pd.Series(nav_out, index=index, name="nav")
    nav_g = pd.Series(nav_gross_out, index=index, name="nav_gross")
    returns = nav_s.pct_change(fill_method=None).fillna(0.0)
    returns.iloc[0] = nav_out[0] / initial_capital - 1.0
    returns_gross = nav_g.pct_change(fill_method=None).fillna(0.0)
    returns_gross.iloc[0] = nav_gross_out[0] / initial_capital - 1.0

    trades = pd.DataFrame(
        trade_rows, columns=["date", "symbol", "delta_weight", "delta_value", "cost"]
    )
    cfg = {
        "rebalance": rebalance,
        "execution_lag": execution_lag,
        "initial_capital": initial_capital,
        "cost_model": cost_model.__dict__,
        "symbols": symbols,
        "start": str(index[0].date()),
        "end": str(index[-1].date()),
        "mode": "fractional",
    }
    if config:
        cfg.update(config)

    return BacktestResult(
        nav=nav_s,
        nav_gross=nav_g,
        returns=returns.rename("returns"),
        returns_gross=returns_gross.rename("returns_gross"),
        weights=pd.DataFrame(weights_out, index=index, columns=symbols),
        turnover=pd.Series(turnover_out, index=index, name="turnover"),
        costs=pd.Series(costs_out, index=index, name="costs"),
        trades=trades,
        config=cfg,
    )
