"""
EDA Backend — US Electricity Market
=====================================
Seven analysis modules, each returning tidy DataFrames / dicts
ready for Plotly rendering in views/eda.py.

1. stl_decompose       — STL (daily seasonality, period=24)
2. mstl_decompose      — MSTL (daily + weekly, periods=(24,168))
3. ren_price_corr      — renewable share vs price, correlation by hour
4. peak_demand_profile — top-N% demand hours: calendar + fuel context
5. duck_curve          — CISO net load by hour-of-day, 2014–present
6. spike_characterise  — price spike magnitude, frequency, duration
7. forecast_mape_heatmap — XGBoost expanding-CV MAPE by hour-of-day
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import db
import forecasting as fc


# ── helpers ────────────────────────────────────────────────────────────────────

def _q(sql: str) -> pd.DataFrame | None:
    con = db.get_connection()
    r   = db.run_query(con, sql)
    return None if r.error or r.df is None or r.df.empty else r.df


def _ts(df: pd.DataFrame, col: str = "hour") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. STL Decomposition
# ══════════════════════════════════════════════════════════════════════════════

def load_demand_series(region: str, year: int) -> pd.DataFrame | None:
    df = _q(f"""
        SELECT hour, demand_mwh
        FROM fact_demand
        WHERE region_id = '{region}'
          AND YEAR(hour) = {year}
          AND demand_mwh > 0
        ORDER BY hour
    """)
    return _ts(df) if df is not None else None


def stl_decompose(df: pd.DataFrame, period: int = 24) -> pd.DataFrame | None:
    """STL with daily period. Returns df with trend, seasonal, resid columns."""
    from statsmodels.tsa.seasonal import STL
    s = df.set_index("hour")["demand_mwh"].asfreq("h")
    if s.isna().sum() > len(s) * 0.05:
        s = s.interpolate("time")
    result = STL(s, period=period, robust=True).fit()
    out = df[["hour", "demand_mwh"]].copy()
    out["trend"]    = result.trend.values
    out["seasonal"] = result.seasonal.values
    out["resid"]    = result.resid.values
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. MSTL Decomposition (daily + weekly)
# ══════════════════════════════════════════════════════════════════════════════

def mstl_decompose(df: pd.DataFrame, periods: tuple = (24, 168)) -> pd.DataFrame | None:
    """MSTL with daily (24h) and weekly (168h) seasonalities."""
    from statsmodels.tsa.seasonal import MSTL
    s = df.set_index("hour")["demand_mwh"].asfreq("h")
    if s.isna().sum() > len(s) * 0.05:
        s = s.interpolate("time")
    result = MSTL(s, periods=periods).fit()
    out = df[["hour", "demand_mwh"]].copy()
    out["trend"] = result.trend.values
    seas = result.seasonal
    if hasattr(seas, "columns"):
        out["seasonal_daily"]  = seas.iloc[:, 0].values
        out["seasonal_weekly"] = seas.iloc[:, 1].values if seas.shape[1] > 1 else np.zeros(len(out))
    else:
        out["seasonal_daily"]  = seas.values
        out["seasonal_weekly"] = np.zeros(len(out))
    out["resid"] = result.resid.values
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Renewable vs Price Correlation
# ══════════════════════════════════════════════════════════════════════════════

def load_ren_price(region: str, year: int) -> pd.DataFrame | None:
    df = _q(f"""
        SELECT
            g.hour,
            HOUR(g.hour) AS hour_of_day,
            ROUND(
                SUM(CASE WHEN g.fuel_id IN ('SUN','WND','WAT')
                         THEN g.generation_mwh ELSE 0 END)
                / NULLIF(SUM(g.generation_mwh), 0), 4
            ) AS ren_share,
            ROUND(AVG(p.price_usd_mwh), 2) AS price_usd_mwh,
            CASE MONTH(g.hour)
                WHEN 12 THEN 'Winter' WHEN 1 THEN 'Winter' WHEN 2 THEN 'Winter'
                WHEN  3 THEN 'Spring' WHEN 4 THEN 'Spring' WHEN 5 THEN 'Spring'
                WHEN  6 THEN 'Summer' WHEN 7 THEN 'Summer' WHEN 8 THEN 'Summer'
                ELSE 'Fall'
            END AS season
        FROM fact_generation g
        JOIN fact_prices p
          ON p.hour = g.hour
         AND p.region_id = g.region_id
         AND p.price_type = 'day_ahead'
        WHERE g.region_id = '{region}'
          AND YEAR(g.hour) = {year}
          AND p.price_usd_mwh BETWEEN -50 AND 500
        GROUP BY g.hour
        HAVING ren_share IS NOT NULL
        ORDER BY g.hour
    """)
    return _ts(df) if df is not None else None


def ren_price_hourly_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Mean correlation between ren_share and price, binned by hour-of-day."""
    return (
        df.groupby("hour_of_day")
        .apply(lambda g: g[["ren_share", "price_usd_mwh"]].corr().iloc[0, 1])
        .reset_index()
        .rename(columns={0: "correlation"})
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Peak Demand Profiling
# ══════════════════════════════════════════════════════════════════════════════

def load_peak_profile(region: str, year: int, top_pct: float = 0.01) -> dict | None:
    """
    Returns:
      peaks     — the extreme demand hours with calendar features
      fuel_peak — avg generation mix during peak hours
      fuel_base — avg generation mix during non-peak hours
    """
    cutoff_sql = f"""
        SELECT PERCENTILE_CONT({1 - top_pct}) WITHIN GROUP (ORDER BY demand_mwh)
        FROM fact_demand
        WHERE region_id = '{region}' AND YEAR(hour) = {year} AND demand_mwh > 0
    """
    cutoff_row = _q(cutoff_sql)
    if cutoff_row is None:
        return None
    cutoff = float(cutoff_row.iloc[0, 0])

    peaks = _q(f"""
        SELECT
            hour,
            demand_mwh,
            HOUR(hour)    AS hour_of_day,
            ISODOW(hour)  AS day_of_week,
            MONTH(hour)   AS month,
            CASE MONTH(hour)
                WHEN 12 THEN 'Winter' WHEN 1 THEN 'Winter' WHEN 2 THEN 'Winter'
                WHEN  3 THEN 'Spring' WHEN 4 THEN 'Spring' WHEN 5 THEN 'Spring'
                WHEN  6 THEN 'Summer' WHEN 7 THEN 'Summer' WHEN 8 THEN 'Summer'
                ELSE 'Fall'
            END AS season
        FROM fact_demand
        WHERE region_id = '{region}'
          AND YEAR(hour) = {year}
          AND demand_mwh >= {cutoff}
        ORDER BY demand_mwh DESC
    """)

    fuel_peak = _q(f"""
        SELECT fuel_id, ROUND(AVG(generation_mwh), 1) AS avg_gen_mwh
        FROM fact_generation
        WHERE region_id = '{region}'
          AND YEAR(hour) = {year}
          AND hour IN (
              SELECT hour FROM fact_demand
              WHERE region_id = '{region}' AND demand_mwh >= {cutoff}
          )
        GROUP BY fuel_id ORDER BY avg_gen_mwh DESC
    """)

    fuel_base = _q(f"""
        SELECT fuel_id, ROUND(AVG(generation_mwh), 1) AS avg_gen_mwh
        FROM fact_generation
        WHERE region_id = '{region}'
          AND YEAR(hour) = {year}
          AND hour NOT IN (
              SELECT hour FROM fact_demand
              WHERE region_id = '{region}' AND demand_mwh >= {cutoff}
          )
        GROUP BY fuel_id ORDER BY avg_gen_mwh DESC
    """)

    return {
        "peaks":     _ts(peaks) if peaks is not None else None,
        "fuel_peak": fuel_peak,
        "fuel_base": fuel_base,
        "cutoff_mwh": cutoff,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Duck Curve
# ══════════════════════════════════════════════════════════════════════════════

def load_duck_curve(region: str = "CISO") -> pd.DataFrame | None:
    df = _q(f"""
        SELECT
            YEAR(d.hour)   AS year,
            HOUR(d.hour)   AS hour_of_day,
            MONTH(d.hour)  AS month,
            ROUND(AVG(
                d.demand_mwh
                - COALESCE(sun.generation_mwh, 0)
                - COALESCE(wnd.generation_mwh, 0)
            ), 0) AS net_load_mwh,
            ROUND(AVG(d.demand_mwh), 0)                      AS gross_load_mwh,
            ROUND(AVG(COALESCE(sun.generation_mwh, 0)), 0)   AS solar_mwh,
            ROUND(AVG(COALESCE(wnd.generation_mwh, 0)), 0)   AS wind_mwh
        FROM fact_demand d
        LEFT JOIN fact_generation sun
               ON sun.hour = d.hour
              AND sun.region_id = d.region_id AND sun.fuel_id = 'SUN'
        LEFT JOIN fact_generation wnd
               ON wnd.hour = d.hour
              AND wnd.region_id = d.region_id AND wnd.fuel_id = 'WND'
        WHERE d.region_id = '{region}'
        GROUP BY 1, 2, 3
        ORDER BY 1, 3, 2
    """)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. Price Spike Characterisation
# ══════════════════════════════════════════════════════════════════════════════

def load_spikes_full(region: str, year: int) -> pd.DataFrame | None:
    df = _q(f"""
        WITH stats AS (
            SELECT
                hour,
                price_usd_mwh,
                AVG(price_usd_mwh) OVER w  AS rolling_mean,
                STDDEV(price_usd_mwh) OVER w AS rolling_std
            FROM fact_prices
            WHERE region_id  = '{region}'
              AND price_type = 'day_ahead'
              AND YEAR(hour) = {year}
            WINDOW w AS (ORDER BY hour ROWS BETWEEN 167 PRECEDING AND CURRENT ROW)
        )
        SELECT
            hour,
            ROUND(price_usd_mwh, 2)  AS price_usd_mwh,
            ROUND(rolling_mean, 2)   AS rolling_mean,
            ROUND((price_usd_mwh - rolling_mean) / NULLIF(rolling_std, 0), 2) AS z_score,
            HOUR(hour)   AS hour_of_day,
            MONTH(hour)  AS month,
            ISODOW(hour) AS day_of_week,
            CASE MONTH(hour)
                WHEN 12 THEN 'Winter' WHEN 1 THEN 'Winter' WHEN 2 THEN 'Winter'
                WHEN  3 THEN 'Spring' WHEN 4 THEN 'Spring' WHEN 5 THEN 'Spring'
                WHEN  6 THEN 'Summer' WHEN 7 THEN 'Summer' WHEN 8 THEN 'Summer'
                ELSE 'Fall'
            END AS season
        FROM stats
        WHERE price_usd_mwh > rolling_mean + 3 * rolling_std
        ORDER BY hour
    """)
    return _ts(df) if df is not None else None


def spike_duration_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Label consecutive spike runs and compute duration per run."""
    df = df.sort_values("hour").copy()
    df["gap"] = (df["hour"].diff() > pd.Timedelta("1h")).cumsum()
    runs = (
        df.groupby("gap").agg(
            start=("hour", "first"),
            end=("hour", "last"),
            n_hours=("hour", "count"),
            max_price=("price_usd_mwh", "max"),
            max_z=("z_score", "max"),
            season=("season", "first"),
        )
        .reset_index(drop=True)
    )
    return runs


def spike_monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("month")
        .agg(
            spike_count=("price_usd_mwh", "count"),
            avg_price=("price_usd_mwh", "mean"),
            max_price=("price_usd_mwh", "max"),
        )
        .round(2)
        .reset_index()
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. Forecast Accuracy Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def forecast_mape_heatmap(
    region: str,
    train_years: int = 2,
    n_folds: int = 8,
) -> pd.DataFrame | None:
    """
    Run XGBoost expanding CV, compute MAPE per (hour-of-day, fold).
    Returns long DataFrame for px.density_heatmap or pivot for px.imshow.
    """
    con = db.get_connection()
    r   = db.run_query(con, f"""
        SELECT hour, demand_mwh
        FROM fact_demand
        WHERE region_id = '{region}'
          AND hour >= (SELECT MAX(hour) FROM fact_demand)
                      - INTERVAL '{train_years} years'
          AND demand_mwh > 0
        ORDER BY hour
    """)
    if r.error or r.df is None or r.df.empty:
        return None

    df_raw  = r.df.copy()
    df_raw["hour"] = pd.to_datetime(df_raw["hour"])
    df_feat = fc.engineer_features(df_raw)
    ml_res  = fc.ml_expanding_cv(df_feat, n_folds=n_folds, n_initial_days=60)

    rows = []
    for res in ml_res:
        ts   = pd.DatetimeIndex(res["timestamps"])
        hod  = ts.hour
        act  = res["actual"]
        pred = res["xgb_pred"]
        for h, a, p in zip(hod, act, pred):
            mape = abs(a - p) / max(abs(a), 1.0) * 100
            rows.append({"hour_of_day": int(h), "fold": res["fold"] + 1, "mape": mape})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    pivot = (
        df.groupby("hour_of_day")["mape"]
        .agg(["mean", "median", "std"])
        .round(2)
        .reset_index()
        .rename(columns={"mean": "MAPE_mean", "median": "MAPE_median", "std": "MAPE_std"})
    )
    return pivot
