"""Share-based simulation for a given account size.

Same conventions as the fractional engine (signal at close t, execution at close t+lag,
rebalance dates from the schedule) but positions are whole shares sized on the raw
price, commissions follow the per-order schedule with its minimum, and cash sits idle.

Total-return handling: a position's value compounds with the *adjusted* return of its
symbol (dividends reinvested in the position), while the number of shares is only used
to size trades on the raw price. This is the standard "adjusted shares" approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.backtest.costs import IBKR_DEFAULT, CostModel
from quantlab.backtest.engine import validate_target_weights
from quantlab.backtest.result import BacktestResult
from quantlab.backtest.schedule import Rebalance, rebalance_mask
from quantlab.data.panel import PricePanel
from quantlab.portfolio.lots import discretize_weights


def simulate_real_capital(
    panel: PricePanel,
    target_weights: pd.DataFrame,
    capital: float,
    cost_model: CostModel = IBKR_DEFAULT,
    rebalance: Rebalance = "monthly",
    execution_lag: int = 1,
    config: dict | None = None,
) -> BacktestResult:
    index, symbols = panel.index, panel.symbols
    n_dates, n_assets = len(index), len(symbols)

    targets = validate_target_weights(target_weights, panel)
    signal_mask = rebalance_mask(index, rebalance)
    exec_targets = targets.where(signal_mask, other=np.nan).shift(execution_lag)
    is_exec = exec_targets.notna().any(axis=1).to_numpy()

    rets = panel.returns().to_numpy()
    rets = np.where(np.isnan(rets), 0.0, rets)

    shares = pd.Series(0.0, index=symbols)
    position_value = np.zeros(n_assets)  # adjusted value of each position
    cash = float(capital)
    cash_gross = float(capital)
    pos_gross = np.zeros(n_assets)

    nav_out = np.empty(n_dates)
    nav_gross_out = np.empty(n_dates)
    weights_out = np.zeros((n_dates, n_assets))
    turnover_out = np.zeros(n_dates)
    costs_out = np.zeros(n_dates)
    trade_rows: list[tuple] = []

    for t in range(n_dates):
        position_value *= 1.0 + rets[t]
        pos_gross *= 1.0 + rets[t]
        nav = cash + position_value.sum()
        nav_gross = cash_gross + pos_gross.sum()

        if is_exec[t]:
            target = exec_targets.iloc[t].fillna(0.0)
            prices_raw = panel.close_raw.iloc[t]
            alloc = discretize_weights(target, prices_raw, nav, cost_model, shares)
            old_w = position_value / nav if nav > 0 else np.zeros(n_assets)
            new_value = (alloc.shares * prices_raw.fillna(0.0)).to_numpy(dtype=float)
            # Positions keep their adjusted value when untouched; traded ones are repriced.
            traded = alloc.orders.to_numpy() != 0
            new_pos = np.where(traded, new_value, position_value)
            turnover = float(np.abs(new_pos / nav - old_w).sum()) if nav > 0 else 0.0
            cost_frac = alloc.commissions / nav if nav > 0 else 0.0
            if traded.any():
                cash = nav - new_pos.sum() - alloc.commissions
                position_value = new_pos
                # gross twin: same shares, no commissions
                cash_gross = nav_gross - new_pos.sum() * (nav_gross / nav)
                pos_gross = new_pos * (nav_gross / nav)
                shares = alloc.shares.astype(float)
                turnover_out[t] = turnover
                costs_out[t] = cost_frac
                for j in np.where(traded)[0]:
                    trade_rows.append(
                        (
                            index[t],
                            symbols[j],
                            float(new_pos[j] / nav - old_w[j]),
                            float(alloc.orders.iloc[j] * prices_raw.iloc[j]),
                            float(cost_model.order_commission(int(alloc.orders.iloc[j]))),
                        )
                    )
            nav = cash + position_value.sum()
            nav_gross = cash_gross + pos_gross.sum()

        nav_out[t] = nav
        nav_gross_out[t] = nav_gross
        weights_out[t] = position_value / nav if nav > 0 else 0.0

    nav_s = pd.Series(nav_out, index=index, name="nav")
    nav_g = pd.Series(nav_gross_out, index=index, name="nav_gross")
    returns = nav_s.pct_change(fill_method=None).fillna(0.0)
    returns.iloc[0] = nav_out[0] / capital - 1.0
    returns_gross = nav_g.pct_change(fill_method=None).fillna(0.0)
    returns_gross.iloc[0] = nav_gross_out[0] / capital - 1.0
    cfg = {
        "rebalance": rebalance,
        "execution_lag": execution_lag,
        "initial_capital": capital,
        "cost_model": cost_model.__dict__,
        "symbols": symbols,
        "start": str(index[0].date()),
        "end": str(index[-1].date()),
        "mode": "real_capital",
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
        trades=pd.DataFrame(
            trade_rows, columns=["date", "symbol", "delta_weight", "delta_value", "cost"]
        ),
        config=cfg,
    )
