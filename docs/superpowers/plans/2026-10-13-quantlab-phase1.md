# Quantlab Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete phase-1 research stack: free daily data pipeline, home-grown vectorized backtest engine validated against vectorbt, two ETF strategies, anti-overfitting validation layer, experiment registry, HTML reports, and a first documented research run.

**Architecture:** One Python package `quantlab` in strict layers (data → features → strategies → backtest → validation/portfolio → research/reports → cli). Strategies output target weights only; the engine applies the t → t+1 execution lag. Every backtest is a pure function of a hashed config and immutable Parquet data.

**Tech Stack:** Python 3.12 via `uv`, pandas 3, numpy 2, pyarrow, duckdb, pydantic 2, pyyaml, plotly, quantstats, statsmodels, scikit-learn, yfinance; dev: pytest, pytest-cov, ruff, vectorbt (test oracle only).

## Global Constraints

- Python `>=3.12`, dependencies managed with `uv` (already in `pyproject.toml`).
- Code, identifiers and docstrings in English (public GitHub repo). Documentation under `docs/` and report labels in French (support de cours).
- Daily frequency only. Signal at close t, execution at close t+1 (`execution_lag=1` default). The engine applies the lag; strategies never do.
- Long only, weights in [0, 1], row sum ≤ 1, cash is the remainder.
- No network calls in the test suite. Network integration tests are marked `@pytest.mark.network` and deselected by default.
- `data/`, `research/experiments.duckdb`, `reports/output/` are gitignored.
- Every run requires a registered hypothesis with a written economic rationale.
- Commit after every task with a conventional-commit message ending in `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File Structure

```
quantlab/
  __init__.py
  config.py                 BacktestConfig (pydantic) + config_hash()
  cli.py                    argparse: fetch, run, report
  data/
    __init__.py
    panel.py                PricePanel dataclass + make_synthetic_panel()
    universe.py             Universe, UniverseSymbol, load_universe()
    universes/etf_core.yaml
    quality.py              check_panel() -> list[Anomaly], DataQualityError
    store.py                PriceStore: parquet files + duckdb metadata + panel()
    providers/
      __init__.py
      base.py               PriceProvider protocol, normalize_ohlcv()
      yfinance_provider.py  YFinanceProvider
  features/
    __init__.py
    returns.py              simple_returns, log_returns
    momentum.py             momentum(close, lookback, skip)
    volatility.py           realized_volatility(returns, window)
    trend.py                sma, above_sma
  strategies/
    __init__.py
    base.py                 Strategy ABC
    registry.py             REGISTRY dict + get_strategy()
    equal_weight.py         EqualWeight
    dual_momentum.py        DualMomentum
  backtest/
    __init__.py
    costs.py                CostModel
    schedule.py             rebalance_mask()
    result.py               BacktestResult
    engine.py               run_backtest()
    benchmarks.py           buy_and_hold_weights(), equal_weight_weights()
  validation/
    __init__.py
    metrics.py              Metrics dataclass, compute_metrics()
    causality.py            assert_causal()
    sensitivity.py          grid_sweep()
    walk_forward.py         walk_forward()
    deflated_sharpe.py      probabilistic_sharpe_ratio(), deflated_sharpe_ratio()
    bootstrap.py            block_bootstrap_sharpe()
  portfolio/
    __init__.py
    lots.py                 discretize_weights()
    real_capital.py         simulate_real_capital()
  research/
    __init__.py
    registry.py             ExperimentRegistry (duckdb)
  reports/
    __init__.py
    tearsheet.py            build_report() -> HTML (plotly)
tests/
  conftest.py               synthetic panel fixtures
  test_panel.py test_universe.py test_providers.py test_store.py test_quality.py
  test_features.py test_schedule.py test_costs.py test_engine.py test_parity_vectorbt.py
  test_strategies.py test_metrics.py test_walk_forward.py test_deflated_sharpe.py
  test_bootstrap.py test_registry.py test_lots.py test_real_capital.py test_report.py test_config.py
docs/
  research/journal.md       journal de recherche (hypothèses, runs, conclusions)
README.md
```

---

### Task 1: Project scaffold

**Files:**
- Create: `quantlab/__init__.py`, all subpackage `__init__.py` files listed above (empty)
- Create: `tests/conftest.py`, `tests/test_smoke.py`
- Modify: `pyproject.toml` (ruff, pytest config, package build)
- Create: `README.md`

**Interfaces:**
- Produces: `tests/conftest.py` fixtures `synthetic_panel` (used by nearly every later test) — defined after Task 2 lands; in this task only the smoke test.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_smoke.py
def test_import():
    import quantlab
    assert quantlab.__version__ == "0.1.0"
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_smoke.py -v` → FAIL (no module `quantlab`)

- [ ] **Step 3: Create the package skeleton and config**

`quantlab/__init__.py`:
```python
"""Quantlab: a personal quantitative research and backtesting stack."""
__version__ = "0.1.0"
```

Empty `__init__.py` in `quantlab/{data,data/providers,features,strategies,backtest,validation,portfolio,research,reports}`.

Append to `pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["quantlab"]

[project.scripts]
quantlab = "quantlab.cli:main"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not network'"
markers = ["network: tests hitting external APIs (deselected by default)"]
```

Set `description = "Personal quantitative research and backtesting stack"` in `[project]`.

- [ ] **Step 4: Run, expect PASS** — `uv sync && uv run pytest -v`

- [ ] **Step 5: README skeleton** (French, sections: Objectif, Principes, Installation, Structure, Règle d'or de la recherche) and commit `chore: project scaffold`.

---

### Task 2: PricePanel and synthetic data

**Files:**
- Create: `quantlab/data/panel.py`, `tests/test_panel.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class PricePanel:
      close: pd.DataFrame; open: pd.DataFrame; high: pd.DataFrame
      low: pd.DataFrame; volume: pd.DataFrame; close_raw: pd.DataFrame
      @property symbols -> list[str]; @property index -> pd.DatetimeIndex
      def returns(self) -> pd.DataFrame            # close.pct_change(), first row NaN
      def slice(self, start=None, end=None) -> PricePanel
      def until(self, date) -> PricePanel          # rows <= date
      def with_close(self, close: pd.DataFrame) -> PricePanel  # for perturbation tests
  def make_synthetic_panel(symbols, start="2015-01-01", periods=1500, seed=0,
                           drift=0.0003, vol=0.01) -> PricePanel
  ```
  `make_synthetic_panel` uses business-day index, geometric random walk, `open=high=low=close`, `volume=1e6`, `close_raw=close`.

- [ ] **Step 1: Tests**

```python
# tests/test_panel.py
import numpy as np, pandas as pd, pytest
from quantlab.data.panel import PricePanel, make_synthetic_panel

def test_synthetic_panel_shapes():
    p = make_synthetic_panel(["A", "B"], periods=100, seed=1)
    assert p.symbols == ["A", "B"]
    assert len(p.index) == 100
    assert p.close.shape == p.close_raw.shape == (100, 2)
    assert (p.close > 0).all().all()

def test_returns_first_row_nan():
    p = make_synthetic_panel(["A"], periods=10)
    r = p.returns()
    assert np.isnan(r.iloc[0, 0]) and r.shape == (10, 1)

def test_until_and_slice():
    p = make_synthetic_panel(["A"], periods=50)
    d = p.index[9]
    assert len(p.until(d).index) == 10
    assert len(p.slice(start=p.index[10], end=p.index[19]).index) == 10

def test_mismatched_columns_rejected():
    p = make_synthetic_panel(["A", "B"], periods=5)
    with pytest.raises(ValueError):
        PricePanel(close=p.close[["A"]], open=p.open, high=p.high, low=p.low,
                   volume=p.volume, close_raw=p.close_raw)
```

- [ ] **Step 2: Run → FAIL**  
- [ ] **Step 3: Implement `quantlab/data/panel.py`** per the interface; `__post_init__` checks all six frames share index and columns, index is monotonic increasing `DatetimeIndex`.  
- [ ] **Step 4: Add fixture to `tests/conftest.py`:**

```python
import pytest
from quantlab.data.panel import make_synthetic_panel

@pytest.fixture
def synthetic_panel():
    return make_synthetic_panel(["AAA", "BBB", "CCC", "DDD"], periods=1500, seed=42)
```

- [ ] **Step 5: Run → PASS, commit `feat(data): PricePanel and synthetic generator`**

---

### Task 3: Universe definitions

**Files:**
- Create: `quantlab/data/universe.py`, `quantlab/data/universes/etf_core.yaml`, `tests/test_universe.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class UniverseSymbol: symbol: str; inception: date; description: str; asset_class: str
  @dataclass(frozen=True)
  class Universe:
      name: str; description: str; survivorship_bias: bool; symbols: tuple[UniverseSymbol, ...]
      @property tickers -> list[str]
      def active_on(self, d: date) -> list[str]
  def load_universe(name: str) -> Universe      # from package yaml dir
  def load_universe_file(path: Path) -> Universe
  ```

`etf_core.yaml` (survivorship_bias: false), symbols with real inception dates: SPY 1993-01-29, QQQ 1999-03-10, IWM 2000-05-26, EFA 2001-08-17, EEM 2003-04-11, VNQ 2004-09-29, TLT 2002-07-30, IEF 2002-07-30, LQD 2002-07-26, HYG 2007-04-11, GLD 2004-11-18, DBC 2006-02-03, SHY 2002-07-26, BIL 2007-05-30.

- [ ] **Step 1: Tests** — load `etf_core`, assert 14 symbols, `active_on(date(2000,1,1))` contains SPY and QQQ but not IWM; missing file raises `FileNotFoundError`; a yaml with duplicate tickers raises `ValueError`.
- [ ] **Step 2–5:** FAIL → implement → PASS → commit `feat(data): universe loader and etf_core`.

---

### Task 4: Price providers

**Files:**
- Create: `quantlab/data/providers/base.py`, `quantlab/data/providers/yfinance_provider.py`, `tests/test_providers.py`

**Interfaces:**
- Produces:
  ```python
  OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "close_raw"]
  class PriceProvider(Protocol):
      name: str
      def fetch(self, symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]: ...
  def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame
      # input columns: Open High Low Close "Adj Close" Volume (yfinance auto_adjust=False)
      # output: OHLCV_COLUMNS, tz-naive DatetimeIndex named "date", float64, sorted, deduped
      # close = Adj Close; factor = Adj Close / Close; open/high/low *= factor; close_raw = Close
  class YFinanceProvider: name = "yfinance"; fetch(...) downloads one symbol at a time
  ```

- [ ] **Step 1: Tests** on a hand-built raw frame: factor applied to open/high/low, `close_raw` untouched, duplicate index rows dropped (keep last), tz-aware index becomes naive, output dtypes float64. Plus `@pytest.mark.network` test fetching SPY for 10 days.
- [ ] **Step 2–5:** FAIL → implement → PASS → commit `feat(data): provider protocol and yfinance adapter`.

---

### Task 5: PriceStore and data quality

**Files:**
- Create: `quantlab/data/store.py`, `quantlab/data/quality.py`, `tests/test_store.py`, `tests/test_quality.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass class Anomaly: symbol: str; date: pd.Timestamp | None; kind: str; detail: str
  class DataQualityError(Exception): anomalies: list[Anomaly]
  def check_panel(panel: PricePanel, max_gap_days: int = 5, max_abs_return: float = 0.5) -> list[Anomaly]
      # kinds: "duplicate_date", "gap", "extreme_return", "non_positive_price"
  class PriceStore:
      def __init__(self, root: Path)             # creates root/raw and root/meta.duckdb
      def save(self, provider: str, symbol: str, df: pd.DataFrame) -> None   # merges with existing, never overwrites older rows
      def load(self, provider: str, symbol: str) -> pd.DataFrame | None
      def update(self, provider: PriceProvider, symbols: list[str], end: date, start: date = date(1990,1,1)) -> None
      def metadata(self) -> pd.DataFrame        # provider, symbol, start, end, rows, fetched_at
      def panel(self, symbols: list[str], start: date | None = None, end: date | None = None,
                provider: str = "yfinance", strict: bool = True) -> PricePanel
      # strict=True raises DataQualityError if check_panel finds anomalies
  ```
  `panel()` aligns on the union of dates, forward-fills nothing (NaN stays NaN before inception), and drops dates where every symbol is NaN.

- [ ] **Step 1: Tests** with `tmp_path` and a `FakeProvider` returning synthetic frames: save/load roundtrip; merge keeps existing rows and appends new ones; metadata rows; `panel()` returns aligned PricePanel with NaN before a late-starting symbol; quality: synthetic gap of 10 business days detected, +80 % return detected, strict mode raises.
- [ ] **Step 2–5:** FAIL → implement → PASS → commit `feat(data): parquet store, duckdb metadata, quality checks`.

---

### Task 6: Features

**Files:**
- Create: `quantlab/features/{returns,momentum,volatility,trend}.py`, `tests/test_features.py`

**Interfaces:**
- Produces (all pure, causal, DataFrame in → DataFrame out, same shape):
  ```python
  def simple_returns(close) -> DataFrame; def log_returns(close) -> DataFrame
  def momentum(close, lookback: int = 252, skip: int = 21) -> DataFrame   # close.shift(skip)/close.shift(lookback) - 1
  def realized_volatility(returns, window: int = 63, annualize: bool = True) -> DataFrame
  def sma(close, window: int) -> DataFrame; def above_sma(close, window: int) -> DataFrame[bool]
  ```

- [ ] **Step 1: Tests** on deterministic series (e.g. close = 1,2,3,…): momentum(lookback=3, skip=1) at row 3 equals close[2]/close[0]-1; NaN for the first `lookback` rows; realized vol of constant returns is 0; `above_sma` is False while below.
- [ ] **Step 2–5:** → commit `feat(features): returns, momentum, volatility, trend`.

---

### Task 7: Rebalance schedule and cost model

**Files:**
- Create: `quantlab/backtest/schedule.py`, `quantlab/backtest/costs.py`, `tests/test_schedule.py`, `tests/test_costs.py`

**Interfaces:**
- Produces:
  ```python
  Rebalance = Literal["daily", "weekly", "monthly"]
  def rebalance_mask(index: pd.DatetimeIndex, freq: Rebalance) -> pd.Series[bool]
      # monthly: last index date of each calendar month; weekly: last index date of each ISO week
  @dataclass(frozen=True)
  class CostModel:
      commission_per_share: float = 0.005; min_commission: float = 1.0; slippage_bps: float = 5.0
      def turnover_cost_rate(self, prices: pd.Series) -> pd.Series
          # per-symbol fractional cost per unit of traded notional:
          # slippage_bps/1e4 + commission_per_share / price
      def order_commission(self, shares: int) -> float    # max(min_commission, |shares|*commission_per_share), 0 if shares == 0
  ZERO_COST = CostModel(0.0, 0.0, 0.0)
  ```

- [ ] **Step 1: Tests** — monthly mask on a Jan–Mar business-day index has exactly 3 True at the last business day of each month; daily mask all True; cost rate at price 100 with 5 bps and 0.005/share = 0.0005 + 0.00005; order_commission(50) == 1.0, (300) == 1.5, (0) == 0.
- [ ] **Step 2–5:** → commit `feat(backtest): rebalance schedule and IBKR-like cost model`.

---

### Task 8: Backtest engine

**Files:**
- Create: `quantlab/backtest/result.py`, `quantlab/backtest/engine.py`, `quantlab/backtest/benchmarks.py`, `tests/test_engine.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class BacktestResult:
      nav: pd.Series; nav_gross: pd.Series; returns: pd.Series; returns_gross: pd.Series
      weights: pd.DataFrame          # post-trade weights held at each close (risky assets only)
      turnover: pd.Series            # one-way turnover on execution dates, 0 elsewhere
      costs: pd.Series               # cost as fraction of NAV per date
      trades: pd.DataFrame           # date, symbol, delta_weight, delta_value, cost
      config: dict                   # strategy name/params, rebalance, cost model, lag, capital
      @property cash -> pd.Series    # 1 - weights.sum(axis=1)
  def run_backtest(panel: PricePanel, target_weights: pd.DataFrame, cost_model: CostModel = CostModel(),
                   rebalance: Rebalance = "monthly", initial_capital: float = 100_000.0,
                   execution_lag: int = 1, config: dict | None = None) -> BacktestResult
  def buy_and_hold_weights(panel, symbol: str) -> pd.DataFrame      # 1.0 on symbol where price is not NaN
  def equal_weight_weights(panel) -> pd.DataFrame                   # 1/n over symbols with non-NaN close at t
  ```

**Algorithm (daily loop over numpy arrays, explicit and documented):**
1. `rets = panel.returns().fillna(0.0)` (NaN price ⇒ 0 return and weight forced to 0).
2. `signal_mask = rebalance_mask(index, rebalance)`; `exec_targets = target_weights.where(signal_mask).shift(execution_lag)`; rows with any non-NaN value are execution dates. Targets are validated: no negatives, row sum ≤ 1 + 1e-9, NaN treated as 0.
3. State: `w` (vector, risky weights after last close), `nav`, `nav_gross`. For each date t:
   - `r_p = w @ rets[t]`; `nav *= (1 + r_p)`; `nav_gross *= (1 + r_p)`; `w = w * (1 + rets[t]) / (1 + r_p)` (if `1 + r_p == 0`, w = 0).
   - If t is an execution date: `delta = target - w`; `turnover = |delta|.sum()`; `cost = (|delta| * cost_rate[t]).sum()` where `cost_rate = cost_model.turnover_cost_rate(close_raw[t])`; `nav *= (1 - cost)`; record trades; `w = target`.
   - Record nav, weights, turnover, cost.
4. `returns = nav.pct_change().fillna(0)` with nav[0] = initial_capital before any return (first date gets return 0 because no position is held before the first execution).

- [ ] **Step 1: Tests**

```python
def test_constant_return_single_asset_no_cost(): # close grows 1%/day, buy&hold weights, daily rebalance, zero cost
    # nav after first execution follows (1.01)**k exactly
def test_lag_applied(): # target 1.0 on asset from first date; weights.iloc[0] == 0, weights.iloc[1] == 1
def test_fifty_fifty_turnover_and_drift(): # two assets, one flat, one +10% day; monthly rebalance; check drifted weights 0.5*1.1/1.05 and turnover at next rebalance == 2*|0.5-0.5238|
def test_costs_reduce_nav(): # same run with CostModel(0,0,10 bps): nav_net < nav_gross by exactly turnover*10bps on first trade
def test_rejects_negative_or_leveraged_weights()
def test_nan_prices_force_zero_weight()
```

- [ ] **Step 2–5:** → commit `feat(backtest): vectorized daily engine with costs and execution lag`.

---

### Task 9: Parity with vectorbt

**Files:**
- Create: `tests/test_parity_vectorbt.py`

Three cases on `synthetic_panel` with `ZERO_COST`: (a) buy-and-hold one symbol, (b) equal-weight monthly, (c) weights that change every month from a seeded random generator. Reference built with `vbt.Portfolio.from_orders(close, size=target_weights_exec, size_type="targetpercent", group_by=True, cash_sharing=True, call_seq="auto", init_cash=100_000, freq="D")` where `target_weights_exec` is the lagged target on execution dates (NaN elsewhere). Assert `np.allclose(result.nav, pf.value(), rtol=1e-8)`. A fourth case with `slippage_bps=10` and `commission_per_share=0` compared against vectorbt `slippage=0.001` with tolerance 1e-6 (document in the test why not exact: vectorbt applies slippage to price, we apply to notional).

- [ ] Run, fix any engine discrepancy at the source (never relax the tolerance), commit `test(backtest): parity with vectorbt`.

---

### Task 10: Strategies and causality test

**Files:**
- Create: `quantlab/strategies/base.py`, `registry.py`, `equal_weight.py`, `dual_momentum.py`, `quantlab/validation/causality.py`, `tests/test_strategies.py`

**Interfaces:**
```python
class Strategy(ABC):
    name: ClassVar[str]
    def __init__(self, **params): ...                        # stored in self.params (dict)
    @abstractmethod
    def target_weights(self, panel: PricePanel) -> pd.DataFrame: ...
    @classmethod
    def param_grid_default(cls) -> dict[str, list]: ...     # used by sensitivity/walk-forward
class EqualWeight(Strategy): name = "equal_weight"            # 1/n over symbols with price at t
class DualMomentum(Strategy): name = "dual_momentum"
    # params: lookback=252, skip=21, top_n=3, absolute_filter=True, safe_asset: str | None = None
    # score = momentum(close, lookback, skip); rank descending among non-NaN; pick top_n;
    # if absolute_filter: keep only picks with score > 0 (or > safe_asset score when given);
    # equal weight among kept picks; remainder → safe_asset if given else cash.
REGISTRY: dict[str, type[Strategy]]; def get_strategy(name) -> type[Strategy]
def assert_causal(strategy: Strategy, panel: PricePanel, t: pd.Timestamp, seed: int = 0) -> None
    # perturb close (and close_raw) strictly after t with random noise, recompute, assert weights up to t equal (atol 1e-12)
```

- [ ] Tests: equal weight sums to 1 and 1/n; dual momentum picks the asset with highest trailing return in a constructed panel; absolute filter yields all-cash when every momentum is negative; `safe_asset` receives the remainder; parametrized causality test over `REGISTRY` values at three dates.
- [ ] → commit `feat(strategies): equal weight, dual momentum, causality guard`.

---

### Task 11: Metrics

**Files:** `quantlab/validation/metrics.py`, `tests/test_metrics.py`

```python
@dataclass(frozen=True)
class Metrics:
    cagr: float; volatility: float; sharpe: float; sortino: float; max_drawdown: float
    max_drawdown_duration_days: int; calmar: float; skew: float; kurtosis: float
    annual_turnover: float | None; annual_cost_bps: float | None; n_days: int
    def as_dict(self) -> dict
def compute_metrics(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0,
                    turnover: pd.Series | None = None, costs: pd.Series | None = None) -> Metrics
def drawdown_series(nav_or_returns: pd.Series) -> pd.Series
```
Tests: constant 0.01 daily return → vol 0, sharpe inf guarded (return `np.inf`); a series +10 %, −10 % → max drawdown −10 %; known drawdown duration; Sharpe of N(0.0004, 0.01) sample ≈ 0.04*sqrt(252) within tolerance.

- [ ] → commit `feat(validation): performance metrics`.

---

### Task 12: Sensitivity sweep and walk-forward

**Files:** `quantlab/validation/sensitivity.py`, `quantlab/validation/walk_forward.py`, tests

```python
def grid_sweep(panel, strategy_cls, grid: dict[str, list], cost_model, rebalance, score: str = "sharpe") -> pd.DataFrame
    # one row per param combination: params..., metrics.as_dict()
@dataclass
class WalkForwardWindow: train_start; train_end; test_start; test_end; best_params: dict; train_score: float; test_metrics: Metrics
@dataclass
class WalkForwardResult: windows: list[WalkForwardWindow]; oos_returns: pd.Series; oos_metrics: Metrics; n_trials: int
def walk_forward(panel, strategy_cls, grid, cost_model, rebalance, train_years: int = 5, test_years: int = 1,
                 score: str = "sharpe", warmup_days: int = 300) -> WalkForwardResult
    # windows anchored on calendar years; strategy fitted on [train_start - warmup, train_end] but scored on train only;
    # test run on [test_start - warmup, test_end] scored on test only; oos_returns = concat of test slices; n_trials = len(grid combos) * len(windows)
```
Tests on synthetic panel with a tiny grid: number of windows matches the span; oos_returns index is exactly the union of test periods with no overlap; best_params ∈ grid; `n_trials` correct.

- [ ] → commit `feat(validation): grid sweep and walk-forward`.

---

### Task 13: Deflated Sharpe and bootstrap

**Files:** `quantlab/validation/deflated_sharpe.py`, `quantlab/validation/bootstrap.py`, tests

```python
def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n: int, skew: float, kurtosis: float) -> float
    # PSR = Φ( (sr - sr*) * sqrt(n-1) / sqrt(1 - skew*sr + (kurt-1)/4 * sr**2) ), sr per period (not annualized), kurtosis non-excess
def expected_max_sharpe(n_trials: int, sr_variance: float) -> float
    # E[max] ≈ sqrt(var) * ((1-γ) Φ^-1(1 - 1/N) + γ Φ^-1(1 - 1/(N e))), γ = 0.5772156649
def deflated_sharpe_ratio(sr: float, n_trials: int, sr_variance: float, n: int, skew: float, kurtosis: float) -> float
    # PSR with sr_benchmark = expected_max_sharpe(n_trials, sr_variance)
def block_bootstrap_sharpe(returns: pd.Series, block: int = 21, n_boot: int = 1000, seed: int = 0,
                           periods_per_year: int = 252, ci: float = 0.95) -> tuple[float, float, np.ndarray]
```
Tests: PSR with sr == benchmark is 0.5; PSR increases with n; expected max Sharpe increases with N and is 0 for N = 1 with variance 0; DSR of a true-zero strategy with 100 trials is below 0.5; bootstrap CI contains the point estimate and narrows with longer series.

- [ ] → commit `feat(validation): deflated Sharpe and block bootstrap`.

---

### Task 14: Config and experiment registry

**Files:** `quantlab/config.py`, `quantlab/research/registry.py`, tests

```python
class CostConfig(BaseModel): commission_per_share: float = 0.005; min_commission: float = 1.0; slippage_bps: float = 5.0
class BacktestConfig(BaseModel):
    strategy: str; params: dict = {}; universe: str; start: date; end: date
    rebalance: Rebalance = "monthly"; cost: CostConfig = CostConfig(); initial_capital: float = 100_000; execution_lag: int = 1
    def hash(self) -> str                                  # sha256 of canonical json, 12 chars
    def cost_model(self) -> CostModel
class ExperimentRegistry:
    def __init__(self, path: Path)                          # duckdb; tables hypotheses(id, name, rationale, created_at), runs(id, hypothesis_id, config_hash, strategy, params_json, universe, start, end, metrics_json, report_path, created_at)
    def register_hypothesis(self, name: str, rationale: str) -> int   # rationale must be ≥ 50 chars else ValueError
    def log_run(self, hypothesis_id: int, config: BacktestConfig, metrics: Metrics, report_path: str | None = None) -> int
    def n_trials(self, hypothesis_id: int) -> int
    def runs(self, hypothesis_id: int | None = None) -> pd.DataFrame
    def hypotheses(self) -> pd.DataFrame
```
Tests: same config → same hash, param change → different hash; short rationale rejected; two runs → `n_trials == 2`; runs DataFrame has metrics columns unpacked.

- [ ] → commit `feat(research): typed config with hash and duckdb experiment registry`.

---

### Task 15: Real-capital simulation (integer lots)

**Files:** `quantlab/portfolio/lots.py`, `quantlab/portfolio/real_capital.py`, tests

```python
@dataclass
class LotAllocation: shares: pd.Series; weights: pd.Series; cash: float; commissions: float
def discretize_weights(target: pd.Series, prices_raw: pd.Series, capital: float, cost_model: CostModel,
                       current_shares: pd.Series | None = None) -> LotAllocation
    # shares = floor(target * capital / price); commissions on |shares - current| via cost_model.order_commission
def simulate_real_capital(panel, target_weights, capital: float, cost_model=CostModel(), rebalance="monthly",
                          execution_lag: int = 1, config: dict | None = None) -> BacktestResult
    # share-based loop: holdings in shares valued at close (adjusted total-return proxy: position value compounds
    # with adjusted returns; lot size uses close_raw); cash earns 0; commissions per order; slippage on notional
```
Tests: capital 1 000 with a 500 $ stock and target 1.0 → 2 shares, cash 0 with zero commission; with min commission 1 → cash reduced; a 3 000 € run on the synthetic panel has `annual_cost_bps` higher than the fractional run and ends with cash ≥ 0 always.

- [ ] → commit `feat(portfolio): integer-lot real capital simulation`.

---

### Task 16: HTML reports

**Files:** `quantlab/reports/tearsheet.py`, `tests/test_report.py`

```python
def build_report(result: BacktestResult, benchmark: BacktestResult | None, title: str, out_path: Path,
                 extra: dict | None = None) -> Path
    # self-contained HTML (plotly include_plotlyjs="cdn"): header with config, metrics table (French labels),
    # NAV log-scale vs benchmark, drawdowns, stacked weights area, turnover bars, monthly returns heatmap;
    # extra sections: walk-forward (oos curve + window boundaries), sensitivity heatmap, DSR/bootstrap box when provided in extra
def build_comparison(results: dict[str, BacktestResult], out_path: Path) -> Path
```
Tests: file written, contains title, contains "Sharpe", size > 10 kB; comparison with two results lists both names.

- [ ] → commit `feat(reports): plotly tearsheet and comparison page`.

---

### Task 17: CLI, first real run, research journal

**Files:** `quantlab/cli.py`, `docs/research/journal.md`, `configs/dual_momentum_etf.yaml`, `README.md`

CLI (argparse):
- `quantlab fetch --universe etf_core [--provider yfinance] [--end YYYY-MM-DD]` → `PriceStore("data").update(...)`, prints metadata table.
- `quantlab run --config configs/x.yaml --hypothesis-id N [--walk-forward] [--sweep] [--capital 3000]` → loads panel, builds strategy from registry, runs backtest + equal-weight and SPY benchmarks, optional walk-forward/sweep, DSR + bootstrap, logs the run, writes `reports/output/<hash>.html`, prints metrics.
- `quantlab hypothesis add --name ... --rationale "..."` and `quantlab hypothesis list`.

Config example `configs/dual_momentum_etf.yaml`: strategy dual_momentum, params lookback 252 skip 21 top_n 3 absolute_filter true safe_asset IEF, universe etf_core, start 2008-01-01, end today, rebalance monthly.

Journal entry H1 (French): hypothèse économique du momentum cross-sectionnel + filtre de tendance absolue (Jegadeesh & Titman 1993, Moskowitz-Ooi-Pedersen 2012, Antonacci 2014) ; grille fixée AVANT le run : lookback {126, 189, 252}, skip {0, 21}, top_n {1, 2, 3} ; critères de rejet écrits : DSR < 0.9, Sharpe OOS < 0.3, ou fonctionne sur < 50 % de la grille.

- [ ] Steps: CLI tests with a fake store (no network); then run `uv run quantlab fetch --universe etf_core` for real (network, user machine), run H1, fill the journal with the actual numbers, update README with usage. Commit `feat(cli): fetch/run/hypothesis commands + first research run (H1)`.

---

## Self-review

- **Spec coverage:** §3 data → Tasks 2–5; §4 conventions/interface → Tasks 8, 10; §5 engine/costs/benchmarks → Tasks 7–8; §6 validation/registry → Tasks 11–14; §7 tests incl. parity and causality → Tasks 9–10; §8 reports → Task 16; §9 phase-1 items 1–10 → Tasks 1–17; §10 risks documented in README (Task 1/17). `next_open` execution deliberately not implemented (spec says planned, not phase 1).
- **Types:** `PricePanel`, `CostModel`, `Rebalance`, `BacktestResult`, `Metrics`, `Strategy`, `BacktestConfig` names are consistent across tasks.
- **Placeholders:** none; each task names exact files, interfaces, and test assertions.
