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
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

LOCAL_DB = Path("data/electricity.duckdb")
_lock = threading.Lock()

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
        return "MotherDuck"
    return f"Local · {LOCAL_DB}"


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
        with _lock:
            df = con.execute(sql).fetchdf()
        return QueryResult(df, time.perf_counter() - t0, None)
    except Exception as exc:
        return QueryResult(None, time.perf_counter() - t0, str(exc))


# ── Live system helpers ───────────────────────────────────────────────────────

def get_last_updated(con: duckdb.DuckDBPyConnection) -> str:
    """Timestamp of the most recent demand record — displayed as 'data last updated'."""
    try:
        ts = con.execute("SELECT MAX(hour) FROM fact_demand").fetchone()[0]
        return str(ts)[:16] if ts else "—"
    except Exception:
        return "—"


def get_summary_stats(con: duckdb.DuckDBPyConnection) -> dict:
    try:
        row = con.execute("""
            SELECT
                COUNT(DISTINCT region_id),
                MIN(hour)::DATE,
                MAX(hour)::DATE,
                COUNT(*)
            FROM fact_demand
        """).fetchone()
        gen_rows = con.execute("SELECT COUNT(*) FROM fact_generation").fetchone()[0]
        price_rows = con.execute("SELECT COUNT(*) FROM fact_prices").fetchone()[0]
        return {
            "regions":     int(row[0])  if row[0]  else 0,
            "date_from":   str(row[1])  if row[1]  else "—",
            "date_to":     str(row[2])  if row[2]  else "—",
            "demand_rows": int(row[3])  if row[3]  else 0,
            "gen_rows":    int(gen_rows)    if gen_rows    else 0,
            "price_rows":  int(price_rows)  if price_rows  else 0,
        }
    except Exception:
        return {"regions": 0, "date_from": "—", "date_to": "—",
                "demand_rows": 0, "gen_rows": 0, "price_rows": 0}


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
