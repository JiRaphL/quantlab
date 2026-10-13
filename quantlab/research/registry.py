"""Experiment registry: the memory of the research process.

Two tables in a DuckDB file:

* ``hypotheses``: an idea with its economic rationale, written *before* any test.
* ``runs``: every backtest, linked to a hypothesis, with its config hash and metrics.

``n_trials(hypothesis_id)`` is what the deflated Sharpe ratio needs. Nothing is ever
deleted: failed runs are the most valuable ones to keep.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from quantlab.config import BacktestConfig
from quantlab.validation.metrics import Metrics

MIN_RATIONALE_CHARS = 50

_DDL = [
    "CREATE SEQUENCE IF NOT EXISTS hypothesis_seq START 1",
    "CREATE SEQUENCE IF NOT EXISTS run_seq START 1",
    """
    CREATE TABLE IF NOT EXISTS hypotheses (
        id INTEGER PRIMARY KEY DEFAULT nextval('hypothesis_seq'),
        name VARCHAR NOT NULL,
        rationale VARCHAR NOT NULL,
        created_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY DEFAULT nextval('run_seq'),
        hypothesis_id INTEGER NOT NULL,
        config_hash VARCHAR NOT NULL,
        strategy VARCHAR,
        params_json VARCHAR,
        universe VARCHAR,
        start DATE,
        "end" DATE,
        rebalance VARCHAR,
        metrics_json VARCHAR,
        n_trials_declared INTEGER,
        report_path VARCHAR,
        note VARCHAR,
        created_at TIMESTAMP
    )
    """,
]


class ExperimentRegistry:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            for stmt in _DDL:
                con.execute(stmt)

    def _conn(self):
        return duckdb.connect(str(self.path))

    # -- hypotheses -----------------------------------------------------------------

    def register_hypothesis(self, name: str, rationale: str) -> int:
        if len(rationale.strip()) < MIN_RATIONALE_CHARS:
            raise ValueError(
                f"write a real economic rationale (>= {MIN_RATIONALE_CHARS} characters) "
                "before testing anything"
            )
        with self._conn() as con:
            row = con.execute(
                "INSERT INTO hypotheses (name, rationale, created_at) "
                "VALUES (?, ?, ?) RETURNING id",
                [name, rationale.strip(), datetime.now()],
            ).fetchone()
        return int(row[0])

    def hypotheses(self) -> pd.DataFrame:
        with self._conn() as con:
            return con.execute("SELECT * FROM hypotheses ORDER BY id").df()

    def hypothesis(self, hypothesis_id: int) -> dict:
        with self._conn() as con:
            df = con.execute("SELECT * FROM hypotheses WHERE id = ?", [hypothesis_id]).df()
        if df.empty:
            raise KeyError(f"no hypothesis with id {hypothesis_id}")
        return df.iloc[0].to_dict()

    # -- runs -----------------------------------------------------------------------

    def log_run(
        self,
        hypothesis_id: int,
        config: BacktestConfig,
        metrics: Metrics,
        report_path: str | None = None,
        n_trials_declared: int = 1,
        note: str | None = None,
    ) -> int:
        self.hypothesis(hypothesis_id)  # raises if unknown
        with self._conn() as con:
            row = con.execute(
                """
                INSERT INTO runs (hypothesis_id, config_hash, strategy, params_json, universe,
                                  start, "end", rebalance, metrics_json, n_trials_declared,
                                  report_path, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
                """,
                [
                    hypothesis_id,
                    config.hash(),
                    config.strategy,
                    json.dumps(config.params, sort_keys=True),
                    config.universe,
                    config.start,
                    config.end,
                    config.rebalance,
                    json.dumps(metrics.as_dict()),
                    n_trials_declared,
                    report_path,
                    note,
                    datetime.now(),
                ],
            ).fetchone()
        return int(row[0])

    def n_trials(self, hypothesis_id: int) -> int:
        """Total number of backtests charged to a hypothesis (sum of declared trials)."""
        with self._conn() as con:
            row = con.execute(
                "SELECT COALESCE(SUM(n_trials_declared), 0) FROM runs WHERE hypothesis_id = ?",
                [hypothesis_id],
            ).fetchone()
        return int(row[0])

    def runs(self, hypothesis_id: int | None = None) -> pd.DataFrame:
        with self._conn() as con:
            if hypothesis_id is None:
                df = con.execute("SELECT * FROM runs ORDER BY id").df()
            else:
                df = con.execute(
                    "SELECT * FROM runs WHERE hypothesis_id = ? ORDER BY id", [hypothesis_id]
                ).df()
        if df.empty:
            return df
        metrics = pd.DataFrame([json.loads(m) for m in df["metrics_json"]])
        params = pd.DataFrame([json.loads(p) for p in df["params_json"]]).add_prefix("param_")
        return pd.concat(
            [df.drop(columns=["metrics_json", "params_json"]), params, metrics], axis=1
        )
