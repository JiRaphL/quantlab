"""Command line entry points.

quantlab fetch --universe etf_core [--end 2026-10-13]
quantlab hypothesis add --name "H1" --rationale "..."
quantlab hypothesis list
quantlab run --config configs/dual_momentum_etf.yaml --hypothesis 1 [--walk-forward]
             [--sweep] [--capital 3000] [--no-log]
quantlab runs [--hypothesis 1]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.backtest.benchmarks import buy_and_hold_weights, fixed_weights
from quantlab.backtest.engine import run_backtest
from quantlab.config import BacktestConfig
from quantlab.data.store import PriceStore
from quantlab.data.universe import load_universe
from quantlab.portfolio.real_capital import simulate_real_capital
from quantlab.reports.tearsheet import build_report
from quantlab.research.registry import ExperimentRegistry
from quantlab.strategies.equal_weight import EqualWeight
from quantlab.strategies.registry import get_strategy
from quantlab.validation.bootstrap import block_bootstrap_sharpe
from quantlab.validation.deflated_sharpe import (
    annual_to_daily_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from quantlab.validation.metrics import compute_metrics
from quantlab.validation.sensitivity import grid_sweep
from quantlab.validation.walk_forward import walk_forward

DATA_DIR = Path("data")
REGISTRY_PATH = Path("research/experiments.duckdb")
REPORT_DIR = Path("reports/output")


def cmd_fetch(args: argparse.Namespace) -> int:
    from quantlab.data.providers.yfinance_provider import YFinanceProvider

    universe = load_universe(args.universe)
    store = PriceStore(args.data_dir)
    provider = YFinanceProvider()
    end = date.fromisoformat(args.end) if args.end else date.today()
    print(f"Téléchargement de {len(universe.tickers)} symboles ({args.universe}) jusqu'au {end}")
    store.update(provider, universe.tickers, end=end)
    meta = store.metadata()
    meta = meta[meta["symbol"].isin(universe.tickers)]
    print(meta.to_string(index=False))
    return 0


def cmd_hypothesis(args: argparse.Namespace) -> int:
    reg = ExperimentRegistry(args.registry)
    if args.action == "add":
        hid = reg.register_hypothesis(args.name, args.rationale)
        print(f"Hypothèse enregistrée avec l'id {hid}")
        return 0
    df = reg.hypotheses()
    if df.empty:
        print("Aucune hypothèse enregistrée.")
    else:
        pd.set_option("display.max_colwidth", 80)
        print(df[["id", "name", "created_at"]].to_string(index=False))
        for _, row in df.iterrows():
            print(f"\n[{row['id']}] {row['name']}\n{row['rationale']}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    reg = ExperimentRegistry(args.registry)
    df = reg.runs(args.hypothesis)
    if df.empty:
        print("Aucun run.")
        return 0
    cols = [
        c
        for c in [
            "id",
            "hypothesis_id",
            "config_hash",
            "strategy",
            "universe",
            "start",
            "end",
            "sharpe",
            "cagr",
            "max_drawdown",
            "annual_cost_bps",
            "n_trials_declared",
            "report_path",
        ]
        if c in df.columns
    ]
    print(df[cols].round(3).to_string(index=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = BacktestConfig.from_yaml(args.config)
    universe = load_universe(cfg.universe)
    store = PriceStore(args.data_dir)
    panel = store.panel(universe.tickers, cfg.start, cfg.end, provider=cfg.provider)
    cost_model = cfg.cost_model()
    strategy_cls = get_strategy(cfg.strategy)
    strategy = strategy_cls(**cfg.params)
    config_hash = cfg.hash()
    print(f"Run {config_hash} : {strategy!r} sur {cfg.universe} {cfg.start} → {cfg.end}")

    result = run_backtest(
        panel,
        strategy.target_weights(panel),
        cost_model,
        cfg.rebalance,
        cfg.initial_capital,
        cfg.execution_lag,
        config={**strategy.describe(), "config_hash": config_hash},
        rebalance_band=cfg.rebalance_band,
    )
    metrics = compute_metrics(result.returns, turnover=result.turnover, costs=result.costs)

    bench_name = cfg.benchmark_name()
    if isinstance(cfg.benchmark, str):
        bench_weights = buy_and_hold_weights(panel, cfg.benchmark)
    else:
        bench_weights = fixed_weights(panel, cfg.benchmark)
    benchmark = run_backtest(
        panel,
        bench_weights,
        cost_model,
        cfg.rebalance,
        cfg.initial_capital,
        cfg.execution_lag,
    )
    ew = run_backtest(
        panel,
        EqualWeight().target_weights(panel),
        cost_model,
        cfg.rebalance,
        cfg.initial_capital,
        cfg.execution_lag,
    )
    ew_metrics = compute_metrics(ew.returns, turnover=ew.turnover, costs=ew.costs)
    bench_metrics = compute_metrics(
        benchmark.returns, turnover=benchmark.turnover, costs=benchmark.costs
    )

    extra: dict = {
        "groups": {s.symbol: s.asset_class for s in universe.symbols},
        "notes": [
            f"Benchmark équipondéré ({cfg.rebalance}) : Sharpe {ew_metrics.sharpe:.2f}, "
            f"CAGR {ew_metrics.cagr:.1%}, max drawdown {ew_metrics.max_drawdown:.1%}."
        ],
        "warnings": [],
    }
    if universe.survivorship_bias:
        extra["warnings"].append(
            "Univers avec biais de survie : seuls les titres encore cotés sont inclus. "
            "Les performances sont surestimées."
        )

    n_trials = 1
    if args.sweep:
        grid = strategy_cls.param_grid_default()
        fixed = {k: v for k, v in cfg.params.items() if k not in grid}
        print(f"Balayage de grille : {grid}")
        sweep = grid_sweep(
            panel,
            strategy_cls,
            grid,
            cost_model,
            cfg.rebalance,
            fixed_params=fixed,
            rebalance_band=cfg.rebalance_band,
        )
        extra["sweep"] = sweep
        axes = sorted(grid, key=lambda k: -len(grid[k]))[:2]
        if len(axes) == 2:
            extra["sweep_axes"] = (axes[0], axes[1])
        n_trials += len(sweep)
        share_beats_ew = float((sweep["sharpe"] > ew_metrics.sharpe).mean())
        share_beats_bench = float((sweep["sharpe"] > bench_metrics.sharpe).mean())
        extra["notes"].append(
            f"Grille de {len(sweep)} combinaisons : Sharpe médian {sweep['sharpe'].median():.2f}, "
            f"min {sweep['sharpe'].min():.2f}, max {sweep['sharpe'].max():.2f} ; "
            f"{share_beats_ew:.0%} des combinaisons battent le Sharpe de l'équipondéré, "
            f"{share_beats_bench:.0%} celui du benchmark {bench_name}."
        )
    if args.walk_forward:
        grid = strategy_cls.param_grid_default()
        fixed = {k: v for k, v in cfg.params.items() if k not in grid}
        print("Walk-forward...")
        wf = walk_forward(
            panel,
            strategy_cls,
            grid,
            cost_model,
            cfg.rebalance,
            train_years=args.train_years,
            test_years=1,
            fixed_params=fixed,
            rebalance_band=cfg.rebalance_band,
        )
        extra["walk_forward"] = wf
        n_trials += wf.n_trials

    # Statistical robustness of the headline (in-sample) Sharpe.
    sr_daily = annual_to_daily_sharpe(metrics.sharpe)
    if "sweep" in extra:
        sr_var = float(np.var(annual_to_daily_sharpe(extra["sweep"]["sharpe"].to_numpy())))
    else:
        sr_var = (0.5 / np.sqrt(252)) ** 2  # conservative prior: 0.5 annualised sd across trials
    reg = ExperimentRegistry(args.registry)
    prior_trials = reg.n_trials(args.hypothesis) if args.hypothesis else 0
    total_trials = max(1, prior_trials + n_trials)
    dsr = deflated_sharpe_ratio(
        sr_daily, total_trials, sr_var, metrics.n_days, metrics.skew, metrics.kurtosis
    )
    psr = probabilistic_sharpe_ratio(sr_daily, 0.0, metrics.n_days, metrics.skew, metrics.kurtosis)
    extra["dsr"] = {
        "dsr": dsr,
        "psr": psr,
        "n_trials": total_trials,
        "expected_max_annual": expected_max_sharpe(total_trials, sr_var) * np.sqrt(252),
    }
    lo, hi, _ = block_bootstrap_sharpe(result.returns, n_boot=args.n_boot)
    extra["bootstrap"] = (lo, hi)

    if args.capital:
        rc = simulate_real_capital(
            panel,
            strategy.target_weights(panel),
            args.capital,
            cost_model,
            cfg.rebalance,
            cfg.execution_lag,
            config=strategy.describe(),
            rebalance_band=cfg.rebalance_band,
        )
        extra["real_capital"] = rc

    report_path = REPORT_DIR / f"{config_hash}.html"
    title = f"{cfg.strategy} · {cfg.universe}"
    build_report(result, benchmark, title, report_path, bench_name=bench_name, extra=extra)

    print("\nMétriques (net de frais) :")
    for k, v in metrics.as_dict().items():
        print(f"  {k:28s} {v}")
    print(
        f"\nSharpe déflaté : {dsr:.3f} ({total_trials} essais)  |  "
        f"bootstrap 95 % [{lo:.2f}, {hi:.2f}]"
    )
    if "walk_forward" in extra:
        print(
            f"Walk-forward hors échantillon : Sharpe {wf.oos_metrics.sharpe:.2f}, "
            f"CAGR {wf.oos_metrics.cagr:.1%}, max DD {wf.oos_metrics.max_drawdown:.1%}"
        )
    print(f"Rapport : {report_path}")

    if args.hypothesis and not args.no_log:
        run_id = reg.log_run(
            args.hypothesis,
            cfg,
            metrics,
            str(report_path),
            n_trials_declared=n_trials,
            note=args.note,
        )
        print(f"Run {run_id} enregistré sous l'hypothèse {args.hypothesis}")
    elif not args.hypothesis:
        print("Aucune hypothèse fournie (--hypothesis) : run NON enregistré.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantlab")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--registry", default=str(REGISTRY_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="télécharger / mettre à jour les prix d'un univers")
    f.add_argument("--universe", required=True)
    f.add_argument("--end", default=None)
    f.set_defaults(func=cmd_fetch)

    h = sub.add_parser("hypothesis", help="gérer les hypothèses de recherche")
    h.add_argument("action", choices=["add", "list"])
    h.add_argument("--name")
    h.add_argument("--rationale")
    h.set_defaults(func=cmd_hypothesis)

    r = sub.add_parser("run", help="lancer un backtest depuis un fichier de config")
    r.add_argument("--config", required=True)
    r.add_argument("--hypothesis", type=int, default=None)
    r.add_argument("--walk-forward", action="store_true")
    r.add_argument("--train-years", type=int, default=5)
    r.add_argument("--sweep", action="store_true")
    r.add_argument("--capital", type=float, default=None)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--no-log", action="store_true")
    r.add_argument("--note", default=None)
    r.set_defaults(func=cmd_run)

    ru = sub.add_parser("runs", help="lister les runs enregistrés")
    ru.add_argument("--hypothesis", type=int, default=None)
    ru.set_defaults(func=cmd_runs)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
