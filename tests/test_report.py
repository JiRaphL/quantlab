from quantlab.backtest.benchmarks import buy_and_hold_weights
from quantlab.backtest.costs import ZERO_COST, CostModel
from quantlab.backtest.engine import run_backtest
from quantlab.reports.tearsheet import build_comparison, build_report
from quantlab.strategies.dual_momentum import DualMomentum
from quantlab.validation.walk_forward import walk_forward


def test_build_report_writes_html(synthetic_panel, tmp_path):
    panel = synthetic_panel
    s = DualMomentum(lookback=120, skip=0, top_n=2, safe_asset="DDD")
    res = run_backtest(panel, s.target_weights(panel), CostModel(), config=s.describe())
    bench = run_backtest(panel, buy_and_hold_weights(panel, "AAA"), ZERO_COST)
    out = build_report(
        res,
        bench,
        "Test tearsheet",
        tmp_path / "r.html",
        bench_name="AAA",
        extra={
            "dsr": {"dsr": 0.7, "psr": 0.9, "n_trials": 18, "expected_max_annual": 0.8},
            "bootstrap": (0.1, 1.2),
            "groups": {"AAA": "equity", "BBB": "equity", "CCC": "bond", "DDD": "cash"},
            "notes": ["note de test"],
            "warnings": ["avertissement de test"],
        },
    )
    text = out.read_text(encoding="utf-8")
    assert "Test tearsheet" in text and "Ratio de Sharpe" in text
    assert "Sharpe déflaté" in text and "note de test" in text and "avertissement" in text
    assert out.stat().st_size > 10_000


def test_report_with_walk_forward_and_sweep(synthetic_panel, tmp_path):
    panel = synthetic_panel
    grid = {"lookback": [60, 120], "top_n": [1, 2]}
    wf = walk_forward(
        panel,
        DualMomentum,
        grid,
        ZERO_COST,
        train_years=2,
        warmup_days=150,
        fixed_params={"skip": 0},
    )
    from quantlab.validation.sensitivity import grid_sweep

    sweep = grid_sweep(panel, DualMomentum, grid, ZERO_COST, fixed_params={"skip": 0})
    res = run_backtest(panel, DualMomentum(lookback=120, skip=0).target_weights(panel), ZERO_COST)
    out = build_report(
        res,
        None,
        "WF",
        tmp_path / "wf.html",
        extra={"walk_forward": wf, "sweep": sweep, "sweep_axes": ("lookback", "top_n")},
    )
    text = out.read_text(encoding="utf-8")
    assert "walk-forward" in text and "Sensibilité" in text


def test_build_comparison(synthetic_panel, tmp_path):
    panel = synthetic_panel
    a = run_backtest(panel, buy_and_hold_weights(panel, "AAA"), ZERO_COST)
    b = run_backtest(panel, buy_and_hold_weights(panel, "BBB"), ZERO_COST)
    out = build_comparison({"Hold AAA": a, "Hold BBB": b}, tmp_path / "cmp.html")
    text = out.read_text(encoding="utf-8")
    assert "Hold AAA" in text and "Hold BBB" in text
