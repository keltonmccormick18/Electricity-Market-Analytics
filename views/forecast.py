"""
Demand Forecast — Infrastructure Scaffold
==========================================
Feature engineering pipeline and UI are ready.
XGBoost model training hooks in via train_model() / generate_forecast() below.
"""

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import db
from constants import REGIONS, REGION_COLORS

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_recent_demand(region: str, days: int = 60) -> pd.DataFrame | None:
    con = db.get_connection()
    result = db.run_query(con, f"""
        SELECT hour, demand_mwh
        FROM fact_demand
        WHERE region_id = '{region}'
          AND hour >= (SELECT MAX(hour) FROM fact_demand WHERE region_id = '{region}')
                      - INTERVAL '{days} days'
        ORDER BY hour
    """)
    return result.df if not result.error else None


@st.cache_data(ttl=300)
def load_feature_dataset(region: str, train_years: int) -> pd.DataFrame | None:
    """Build the feature matrix that will be fed into XGBoost."""
    con = db.get_connection()
    result = db.run_query(con, f"""
        SELECT
            d.hour,
            d.demand_mwh,
            HOUR(d.hour)        AS hour_of_day,
            ISODOW(d.hour)      AS day_of_week,
            MONTH(d.hour)       AS month,
            YEAR(d.hour)        AS year,
            -- 24h and 168h (1-week) lagged demand
            LAG(d.demand_mwh, 24)  OVER (ORDER BY d.hour) AS lag_24h,
            LAG(d.demand_mwh, 168) OVER (ORDER BY d.hour) AS lag_168h,
            -- 7-day rolling mean
            AVG(d.demand_mwh) OVER (
                ORDER BY d.hour ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
            ) AS rolling_7d_avg
        FROM fact_demand d
        WHERE d.region_id = '{region}'
          AND d.hour >= (SELECT MAX(hour) FROM fact_demand)
                        - INTERVAL '{train_years} years'
        ORDER BY d.hour
    """)
    return result.df if not result.error else None


# ── Model stubs (wire in XGBoost here) ────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    TODO: Train XGBoost regressor on df.
    Features: hour_of_day, day_of_week, month, year, lag_24h, lag_168h, rolling_7d_avg
    Target:   demand_mwh
    Returns:  trained model object
    """
    raise NotImplementedError("XGBoost training not yet implemented.")


def generate_forecast(model, df_features: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """
    TODO: Run trained model over next *horizon_hours* hours.
    Returns DataFrame with columns: hour, predicted_mwh, lower_ci, upper_ci
    """
    raise NotImplementedError("Forecast generation not yet implemented.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings", anchor=False)
    region          = st.selectbox("Region", REGIONS, index=0)
    horizon_hours   = st.slider("Forecast horizon (hours)", 24, 168, 48, step=24)
    train_years     = st.slider("Training window (years)", 1, 10, 3)
    show_features   = st.checkbox("Show feature matrix preview", value=False)

# ── Page ──────────────────────────────────────────────────────────────────────

st.header("Demand Forecast", anchor=False, divider="gray")

st.info(
    "**Infrastructure ready.** "
    "Feature engineering, data loaders, and UI scaffolding are complete. "
    "Connect `train_model()` and `generate_forecast()` in `views/forecast.py` "
    "to activate live forecasting.",
    icon="🔧",
)

# ── Feature engineering plan ──────────────────────────────────────────────────

with st.expander("📐 Feature Engineering Plan", expanded=True):
    col_feat, col_arch = st.columns(2)

    with col_feat:
        st.markdown("**Input features**")
        st.markdown("""
| Feature | Description |
|---|---|
| `hour_of_day` | 0–23 |
| `day_of_week` | ISO 1–7 |
| `month` | 1–12 |
| `year` | Trend proxy |
| `lag_24h` | Demand 24h ago |
| `lag_168h` | Demand 1 week ago |
| `rolling_7d_avg` | 7-day rolling mean |
        """)

    with col_arch:
        st.markdown("**Model architecture**")
        st.markdown("""
- **Algorithm**: XGBoost Regressor
- **Objective**: `reg:squarederror`
- **Train/test split**: 80 / 20 (time-ordered)
- **Confidence intervals**: ±1.64σ of training residuals (90% CI)
- **Retrain cadence**: weekly incremental update
- **Evaluation**: MAE, MAPE on held-out test set
        """)

# ── Historical demand (context) ───────────────────────────────────────────────

st.subheader(f"Historical demand — {region} (last 60 days)", anchor=False)

df_recent = load_recent_demand(region, days=60)

if df_recent is not None and not df_recent.empty:
    fig = px.line(
        df_recent, x="hour", y="demand_mwh",
        labels={"demand_mwh": "Demand (MWh)", "hour": ""},
        template="plotly_dark",
        color_discrete_sequence=[REGION_COLORS.get(region, "#4fc3f7")],
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
else:
    st.warning("No recent demand data found.")

# ── Feature matrix preview ────────────────────────────────────────────────────

if show_features:
    st.subheader("Feature matrix preview", anchor=False)
    with st.spinner("Building feature matrix…"):
        df_feat = load_feature_dataset(region, train_years)
    if df_feat is not None:
        st.caption(f"{len(df_feat):,} rows  ·  {len(df_feat.columns)} features")
        st.dataframe(df_feat.dropna().tail(200), height=300, width="stretch")

# ── Forecast placeholder ──────────────────────────────────────────────────────

st.subheader("Forecast output", anchor=False)

if st.button("▶ Generate Forecast", type="primary"):
    st.warning(
        f"Model not yet trained. "
        f"Implement `train_model()` and `generate_forecast()` in `views/forecast.py` "
        f"to produce a {horizon_hours}h ahead forecast for {region}.",
        icon="⚠️",
    )

    # ── Placeholder chart shows what the output will look like ────────────────
    if df_recent is not None and not df_recent.empty:
        last_actual = df_recent.tail(horizon_hours).copy()
        import numpy as np
        rng = np.random.default_rng(0)
        noise = rng.normal(0, last_actual["demand_mwh"].std() * 0.05, len(last_actual))
        placeholder = last_actual.copy()
        placeholder["predicted"] = (last_actual["demand_mwh"] + noise).clip(lower=0)
        placeholder["lower_ci"]  = placeholder["predicted"] * 0.93
        placeholder["upper_ci"]  = placeholder["predicted"] * 1.07

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=placeholder["hour"], y=placeholder["demand_mwh"],
            name="Actual", line=dict(color="#4fc3f7"),
        ))
        fig2.add_trace(go.Scatter(
            x=placeholder["hour"], y=placeholder["predicted"],
            name="Forecast (placeholder)", line=dict(color="#ff7f0e", dash="dash"),
        ))
        fig2.add_trace(go.Scatter(
            x=pd.concat([placeholder["hour"], placeholder["hour"].iloc[::-1]]),
            y=pd.concat([placeholder["upper_ci"], placeholder["lower_ci"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI",
        ))
        fig2.update_layout(
            template="plotly_dark", height=340,
            margin=dict(t=10, b=10),
            title="Placeholder — replace with real model output",
        )
        st.plotly_chart(fig2, width="stretch")
