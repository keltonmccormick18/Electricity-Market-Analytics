"""
DuckDB connection layer
========================
Supports two backends, selected automatically:
  1. MotherDuck  — when MOTHERDUCK_TOKEN is present in st.secrets or env
  2. Local file  — data/electricity.duckdb (development / offline)

All public functions are safe to call with an empty database; they return
empty results rather than raising.
"""

import os
import time
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

LOCAL_DB = Path("data/electricity.duckdb")

# ── Connection ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to database…")
def get_connection() -> duckdb.DuckDBPyConnection:
    token = _secret("MOTHERDUCK_TOKEN")
    if token:
        os.environ["motherduck_token"] = token
        db = _secret("MOTHERDUCK_DB") or "energy"
        return duckdb.connect(f"md:{db}")
    LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(LOCAL_DB))


def backend_label(con: duckdb.DuckDBPyConnection) -> str:
    try:
        path = con.execute("SELECT current_database()").fetchone()[0]
    except Exception:
        path = ""
    if "md:" in str(path) or _secret("MOTHERDUCK_TOKEN"):
        return "☁️ MotherDuck"
    return f"💾 Local · {LOCAL_DB}"


# ── Schema introspection ──────────────────────────────────────────────────────

def list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    try:
        rows = con.execute("SHOW TABLES").fetchall()
        return sorted(r[0] for r in rows)
    except Exception:
        return []


def table_info(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    """Return (column_name, data_type, nullable) for *table*."""
    try:
        return con.execute(f'DESCRIBE "{table}"').fetchdf()
    except Exception:
        return pd.DataFrame(columns=["column_name", "column_type", "null"])


def row_count(con: duckdb.DuckDBPyConnection, table: str) -> int | None:
    try:
        return con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    except Exception:
        return None


# ── Query execution ───────────────────────────────────────────────────────────

class QueryResult:
    __slots__ = ("df", "elapsed_s", "error", "row_count", "col_count")

    def __init__(
        self,
        df: pd.DataFrame | None,
        elapsed_s: float,
        error: str | None,
    ):
        self.df        = df
        self.elapsed_s = elapsed_s
        self.error     = error
        self.row_count = len(df) if df is not None else 0
        self.col_count = len(df.columns) if df is not None else 0


def run_query(con: duckdb.DuckDBPyConnection, sql: str) -> QueryResult:
    sql = sql.strip()
    if not sql:
        return QueryResult(None, 0.0, "Empty query.")
    t0 = time.perf_counter()
    try:
        df = con.execute(sql).fetchdf()
        return QueryResult(df, time.perf_counter() - t0, None)
    except Exception as exc:
        return QueryResult(None, time.perf_counter() - t0, str(exc))


# ── Auto-chart heuristic ──────────────────────────────────────────────────────

def suggest_chart(df: pd.DataFrame) -> str | None:
    """
    Returns 'line', 'bar', or None based on column types.
    Caller uses this to decide whether to render a chart.
    """
    if df is None or df.empty or len(df.columns) < 2:
        return None
    dtypes = df.dtypes
    has_time = any(
        pd.api.types.is_datetime64_any_dtype(dtypes[c]) for c in df.columns
    )
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(dtypes[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(dtypes[c])]
    if has_time and num_cols:
        return "line"
    if cat_cols and num_cols and len(df) <= 50:
        return "bar"
    return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _secret(key: str) -> str | None:
    """Read from st.secrets first, then os.environ."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key)
