"""Self-contained HTML tearsheets built with plotly. Labels are in French: these pages
double as teaching material.

Colour roles (validated categorical palette, light/dark selected):
strategy = blue, benchmark = neutral grey, drawdown/negative = red, positive = blue.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from quantlab.backtest.result import BacktestResult
from quantlab.validation.metrics import Metrics, compute_metrics, drawdown_series

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GREY = "#8a8984"
RED = "#e34948"
BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"

LABELS_FR = {
    "cagr": ("Rendement annualisé (CAGR)", "pct"),
    "volatility": ("Volatilité annualisée", "pct"),
    "sharpe": ("Ratio de Sharpe", "num"),
    "sortino": ("Ratio de Sortino", "num"),
    "max_drawdown": ("Perte maximale (max drawdown)", "pct"),
    "max_drawdown_duration_days": ("Durée max sous le plus haut (séances)", "int"),
    "calmar": ("Ratio de Calmar", "num"),
    "skew": ("Asymétrie (skew)", "num"),
    "kurtosis": ("Kurtosis", "num"),
    "annual_turnover": ("Rotation annuelle (turnover)", "num"),
    "annual_cost_bps": ("Frais annuels (points de base)", "num"),
    "total_return": ("Rendement total", "pct"),
    "n_days": ("Nombre de séances", "int"),
}


def _fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, float) and np.isinf(value):
        return "∞"
    if kind == "pct":
        return f"{value:+.2%}"
    if kind == "int":
        return f"{int(value)}"
    return f"{value:.2f}"


def _layout(title: str, height: int = 380, **kw) -> dict:
    base = dict(
        title=dict(text=title, font=dict(size=15, color=TEXT), y=0.97, yanchor="top"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT2, size=12),
        margin=dict(l=50, r=20, t=80, b=40),
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
    )
    base.update(kw)
    return base


def _fig_html(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False})


# -- figures ----------------------------------------------------------------------------


def fig_nav(result: BacktestResult, benchmark: BacktestResult | None, bench_name: str) -> go.Figure:
    fig = go.Figure()
    base = result.nav / result.nav.iloc[0]
    fig.add_trace(
        go.Scatter(
            x=base.index, y=base, name="Stratégie (net de frais)", line=dict(color=BLUE, width=2)
        )
    )
    gross = result.nav_gross / result.nav_gross.iloc[0]
    fig.add_trace(
        go.Scatter(
            x=gross.index,
            y=gross,
            name="Stratégie (brut)",
            line=dict(color=BLUE, width=1, dash="dot"),
        )
    )
    if benchmark is not None:
        b = benchmark.nav / benchmark.nav.iloc[0]
        fig.add_trace(go.Scatter(x=b.index, y=b, name=bench_name, line=dict(color=GREY, width=2)))
    fig.update_layout(**_layout("Valeur liquidative (base 1, échelle log)", 420))
    fig.update_yaxes(type="log", tickformat=".2f")
    return fig


def fig_drawdown(result: BacktestResult, benchmark: BacktestResult | None, bench_name: str):
    fig = go.Figure()
    dd = drawdown_series(result.returns)
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            name="Stratégie",
            fill="tozeroy",
            line=dict(color=RED, width=1.5),
            fillcolor="rgba(227,73,72,0.25)",
        )
    )
    if benchmark is not None:
        bd = drawdown_series(benchmark.returns)
        fig.add_trace(
            go.Scatter(x=bd.index, y=bd, name=bench_name, line=dict(color=GREY, width=1.5))
        )
    fig.update_layout(**_layout("Drawdowns (distance au plus haut historique)", 300))
    fig.update_yaxes(tickformat=".0%")
    return fig


def fig_weights(result: BacktestResult, groups: dict[str, str] | None = None) -> go.Figure:
    w = result.weights.copy()
    if groups:
        w = w.T.groupby(lambda s: groups.get(s, "autre")).sum().T
    w["Cash"] = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    # Weekly resampling keeps the file small without changing the picture.
    w = w.resample("ME").last().dropna(how="all")
    palette = SERIES + BLUE_RAMP
    fig = go.Figure()
    for i, col in enumerate(w.columns):
        color = GREY if col == "Cash" else palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=w.index,
                y=w[col],
                name=str(col),
                stackgroup="one",
                mode="lines",
                line=dict(width=0.5, color=color),
                hovertemplate="%{y:.1%}",
            )
        )
    fig.update_layout(**_layout("Allocation dans le temps (poids de fin de mois)", 380))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def fig_turnover(result: BacktestResult) -> go.Figure:
    t = result.turnover[result.turnover > 0]
    fig = go.Figure(
        go.Bar(x=t.index, y=t, name="Turnover", marker=dict(color=BLUE), hovertemplate="%{y:.1%}")
    )
    fig.update_layout(**_layout("Rotation du portefeuille à chaque rebalancement", 260))
    fig.update_yaxes(tickformat=".0%")
    return fig


def fig_monthly_heatmap(returns: pd.Series) -> go.Figure:
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    table = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "r": monthly})
    pivot = table.pivot(index="year", columns="month", values="r").reindex(columns=range(1, 13))
    months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    lim = float(np.nanmax(np.abs(pivot.to_numpy()))) if pivot.notna().any().any() else 0.1
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=months,
            y=[str(y) for y in pivot.index],
            colorscale=[[0, RED], [0.5, "#f0efec"], [1, BLUE]],
            zmin=-lim,
            zmax=lim,
            text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:+.1%}")(pivot.to_numpy()),
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="%{y} %{x}: %{text}",
            colorbar=dict(tickformat="+.0%"),
        )
    )
    fig.update_layout(**_layout("Rendements mensuels", 60 + 26 * len(pivot), hovermode="closest"))
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_walk_forward(wf) -> go.Figure:
    nav = (1.0 + wf.oos_returns).cumprod()
    fig = go.Figure(
        go.Scatter(
            x=nav.index,
            y=nav,
            name="Hors échantillon (walk-forward)",
            line=dict(color=BLUE, width=2),
        )
    )
    for w in wf.windows:
        fig.add_vline(x=w.test_start, line=dict(color=GRID, width=1))
    fig.update_layout(
        **_layout(
            "Courbe hors échantillon : chaque année utilise les paramètres "
            "choisis sur les années précédentes",
            380,
        )
    )
    fig.update_yaxes(type="log", tickformat=".2f")
    return fig


def fig_sensitivity(sweep: pd.DataFrame, x: str, y: str, metric: str = "sharpe") -> go.Figure:
    pivot = sweep.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale=[[0, BLUE_RAMP[0]], [1, BLUE_RAMP[-1]]],
            text=np.round(pivot.to_numpy(), 2),
            texttemplate="%{text}",
            hovertemplate="%{text}",
        )
    )
    fig.update_layout(
        **_layout(
            f"Sensibilité : {metric} selon {x} (colonnes) et {y} (lignes)",
            80 + 40 * len(pivot),
            hovermode="closest",
        )
    )
    fig.update_xaxes(title=x, type="category")
    fig.update_yaxes(title=y, type="category")
    return fig


# -- tables -----------------------------------------------------------------------------


def metrics_table(columns: dict[str, Metrics]) -> str:
    head = "".join(f"<th>{html.escape(name)}</th>" for name in columns)
    rows = []
    for key, (label, kind) in LABELS_FR.items():
        cells = "".join(f"<td>{_fmt(getattr(m, key), kind)}</td>" for m in columns.values())
        rows.append(f"<tr><th scope='row'>{label}</th>{cells}</tr>")
    body = "".join(rows)
    return (
        f"<table class='metrics'><thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def df_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    def cell(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f"<table class='metrics'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# -- pages ------------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; --surface: #fcfcfb; --text: #0b0b0b; --text2: #52514e;
        --line: #e6e5e1; --accent: #2a78d6; }
body { margin: 0; padding: 24px 16px; background: var(--surface); color: var(--text);
       font: 14px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 32px 0 8px; }
.sub { color: var(--text2); margin-bottom: 16px; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 12px 0;
        background: var(--surface); overflow-x: auto; }
table.metrics { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
table.metrics th, table.metrics td { padding: 6px 10px; border-bottom: 1px solid var(--line);
        text-align: right; }
table.metrics th[scope=row], table.metrics thead th:first-child { text-align: left;
        color: var(--text2); font-weight: 500; }
.note { background: #f3f7fd; border-left: 3px solid var(--accent); padding: 8px 12px;
        margin: 8px 0; }
.warn { background: #fdf3f3; border-left: 3px solid #e34948; padding: 8px 12px; margin: 8px 0; }
code { background: #f0efec; padding: 1px 4px; border-radius: 3px; }
"""


def _page(title: str, subtitle: str, body: str) -> str:
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{_CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1><div class="sub">{subtitle}</div>
{body}
</main></body></html>"""


def build_report(
    result: BacktestResult,
    benchmark: BacktestResult | None,
    title: str,
    out_path: Path | str,
    bench_name: str = "Benchmark",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a tearsheet. ``extra`` may contain: ``walk_forward`` (WalkForwardResult),
    ``sweep`` (DataFrame, with ``sweep_axes=(x, y)``), ``dsr`` (dict), ``bootstrap``
    (lo, hi), ``groups`` (symbol -> asset class), ``real_capital`` (BacktestResult),
    ``notes`` (list of str), ``warnings`` (list of str)."""
    extra = extra or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = compute_metrics(result.returns, turnover=result.turnover, costs=result.costs)
    cols = {"Stratégie": m}
    if benchmark is not None:
        cols[bench_name] = compute_metrics(
            benchmark.returns, turnover=benchmark.turnover, costs=benchmark.costs
        )
    if "real_capital" in extra:
        rc = extra["real_capital"]
        cols[f"Capital réel {rc.config.get('initial_capital', 0):,.0f} $"] = compute_metrics(
            rc.returns, turnover=rc.turnover, costs=rc.costs
        )

    cfg = result.config
    subtitle = (
        f"{html.escape(str(cfg.get('strategy', '')))} {html.escape(str(cfg.get('params', '')))} · "
        f"{cfg.get('start')} → {cfg.get('end')} · rebalancement {cfg.get('rebalance')} · "
        f"exécution t+{cfg.get('execution_lag')} · config <code>{cfg.get('config_hash', '')}</code>"
    )
    parts = []
    for w in extra.get("warnings", []):
        parts.append(f"<div class='warn'>{html.escape(w)}</div>")
    for n in extra.get("notes", []):
        parts.append(f"<div class='note'>{html.escape(n)}</div>")

    parts.append("<h2>Métriques</h2><div class='card'>" + metrics_table(cols) + "</div>")

    if "dsr" in extra or "bootstrap" in extra:
        lines = []
        if "dsr" in extra:
            d = extra["dsr"]
            lines.append(
                f"<b>Sharpe déflaté</b> : probabilité {d['dsr']:.1%} que le Sharpe soit réel "
                f"compte tenu de {d['n_trials']} essais (Sharpe max attendu par pure chance : "
                f"{d['expected_max_annual']:.2f} annualisé). "
                f"Sharpe probabiliste vs 0 : {d['psr']:.1%}."
            )
        if "bootstrap" in extra:
            lo, hi = extra["bootstrap"]
            lines.append(
                f"<b>Bootstrap par blocs</b> : intervalle à 95 % du Sharpe [{lo:.2f} ; {hi:.2f}]."
            )
        parts.append(
            "<h2>Robustesse statistique</h2><div class='card'>" + "<br>".join(lines) + "</div>"
        )

    parts.append("<h2>Performance</h2>")
    parts.append(_fig_html(fig_nav(result, benchmark, bench_name)))
    parts.append(_fig_html(fig_drawdown(result, benchmark, bench_name)))
    parts.append(_fig_html(fig_monthly_heatmap(result.returns)))
    parts.append("<h2>Portefeuille</h2>")
    parts.append(_fig_html(fig_weights(result, extra.get("groups"))))
    parts.append(_fig_html(fig_turnover(result)))

    if "walk_forward" in extra:
        wf = extra["walk_forward"]
        parts.append("<h2>Validation walk-forward</h2>")
        parts.append(
            f"<div class='note'>Paramètres choisis chaque année sur les {len(wf.windows)} fenêtres "
            f"d'entraînement précédentes (critère : {wf.score}), puis appliqués à l'année "
            f"suivante. "
            f"Sharpe hors échantillon : <b>{wf.oos_metrics.sharpe:.2f}</b>, CAGR "
            f"{wf.oos_metrics.cagr:.1%}, max drawdown {wf.oos_metrics.max_drawdown:.1%}. "
            f"{wf.n_trials} backtests exécutés pendant la sélection.</div>"
        )
        parts.append(_fig_html(fig_walk_forward(wf)))
        parts.append("<div class='card'>" + df_table(wf.summary()) + "</div>")

    if "sweep" in extra:
        x, y = extra.get("sweep_axes", (None, None))
        parts.append("<h2>Sensibilité aux paramètres</h2>")
        if x and y:
            parts.append(_fig_html(fig_sensitivity(extra["sweep"], x, y)))
        parts.append("<div class='card'>" + df_table(extra["sweep"].round(3)) + "</div>")

    out_path.write_text(_page(title, subtitle, "\n".join(parts)), encoding="utf-8")
    return out_path


def build_comparison(
    results: dict[str, BacktestResult],
    out_path: Path | str,
    title: str = "Comparaison de stratégies",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    cols = {}
    for i, (name, r) in enumerate(results.items()):
        nav = r.nav / r.nav.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=nav.index, y=nav, name=name, line=dict(color=SERIES[i % len(SERIES)], width=2)
            )
        )
        cols[name] = compute_metrics(r.returns, turnover=r.turnover, costs=r.costs)
    fig.update_layout(**_layout("Valeur liquidative (base 1, échelle log)", 460))
    fig.update_yaxes(type="log", tickformat=".2f")
    body = "<div class='card'>" + metrics_table(cols) + "</div>" + _fig_html(fig)
    out_path.write_text(_page(title, f"{len(results)} stratégies", body), encoding="utf-8")
    return out_path
