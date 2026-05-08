import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import db
from constants import FUEL_COLORS, FUEL_LABELS, FUEL_ORDER, REGIONS

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_generation(region: str, date_from: str, date_to: str) -> pd.DataFrame | None:
    con = db.get_connection()
    result = db.run_query(con, f"""
        SELECT
            DATE_TRUNC('month', hour)::DATE AS month,
            fuel_id,
            ROUND(SUM(generation_mwh) / 1000, 1) AS generation_gwh
        FROM fact_generation
        WHERE region_id = '{region}'
          AND hour >= '{date_from}'
          AND hour <  '{date_to}'
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)
    return result.df if not result.error else None


@st.cache_data(ttl=300)
def load_mix_totals(region: str, date_from: str, date_to: str) -> pd.DataFrame | None:
    con = db.get_connection()
    result = db.run_query(con, f"""
        SELECT
            fuel_id,
            ROUND(SUM(generation_mwh) / 1e6, 2) AS total_twh,
            ROUND(100.0 * SUM(generation_mwh) / NULLIF(SUM(SUM(generation_mwh)) OVER (), 0), 1) AS share_pct
        FROM fact_generation
        WHERE region_id = '{region}'
          AND hour >= '{date_from}'
          AND hour <  '{date_to}'
        GROUP BY fuel_id
        ORDER BY total_twh DESC
    """)
    return result.df if not result.error else None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters", anchor=False)
    region = st.selectbox("Region", REGIONS, index=0)

    years = list(range(2014, 2026))
    year_from, year_to = st.select_slider(
        "Year range",
        options=years,
        value=(2019, 2024),
    )
    date_from = f"{year_from}-01-01"
    date_to   = f"{year_to}-12-31"

# ── Page ──────────────────────────────────────────────────────────────────────

st.header(f"Generation Mix — {region}", anchor=False, divider="gray")

df = load_generation(region, date_from, date_to)

if df is None or df.empty:
    st.warning("No generation data for the selected filters.")
    st.stop()

# Map codes → readable labels and apply stack order
df["fuel_label"] = df["fuel_id"].map(FUEL_LABELS).fillna(df["fuel_id"])
label_order = [FUEL_LABELS[f] for f in FUEL_ORDER if FUEL_LABELS[f] in df["fuel_label"].unique()]
label_colors = {FUEL_LABELS[k]: v for k, v in FUEL_COLORS.items()}

# ── Stacked area chart ────────────────────────────────────────────────────────

fig = px.area(
    df,
    x="month",
    y="generation_gwh",
    color="fuel_label",
    color_discrete_map=label_colors,
    category_orders={"fuel_label": label_order},
    labels={"generation_gwh": "Generation (GWh)", "fuel_label": "Fuel", "month": ""},
    template="plotly_dark",
)
fig.update_layout(
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=60, b=20),
    hovermode="x unified",
)
fig.update_traces(line=dict(width=0.3))
st.plotly_chart(fig, width="stretch")

# ── Bottom: donut + table ─────────────────────────────────────────────────────

totals = load_mix_totals(region, date_from, date_to)

col_chart, col_table = st.columns([1, 1])

with col_chart:
    st.subheader("Fuel mix (period total)", anchor=False)
    if totals is not None and not totals.empty:
        totals["fuel_label"] = totals["fuel_id"].map(FUEL_LABELS).fillna(totals["fuel_id"])
        pie = px.pie(
            totals,
            names="fuel_label",
            values="total_twh",
            color="fuel_label",
            color_discrete_map=label_colors,
            hole=0.45,
            template="plotly_dark",
        )
        pie.update_traces(textposition="inside", textinfo="percent+label")
        pie.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(pie, width="stretch")

with col_table:
    st.subheader("Summary table", anchor=False)
    if totals is not None and not totals.empty:
        display = totals[["fuel_id", "total_twh", "share_pct"]].copy()
        display["fuel_id"] = display["fuel_id"].map(FUEL_LABELS).fillna(display["fuel_id"])
        display.columns = ["Fuel", "Total (TWh)", "Share (%)"]
        st.dataframe(display, hide_index=True, width="stretch")
