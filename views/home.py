import streamlit as st

import db

st.title("⚡ Energy Grid Analytics")
st.caption("US electricity market data — EIA Open Data, served live from MotherDuck")

con = db.get_connection()

# ── Live status bar ───────────────────────────────────────────────────────────

last = db.get_last_updated(con)
stats = db.get_summary_stats(con)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Data last updated", last)
c2.metric("Regions", stats["regions"])
c3.metric("Date range", f"{stats['date_from'][:7]} → {stats['date_to'][:7]}")
c4.metric("Demand records", f"{stats['demand_rows']:,}")
c5.metric("Generation records", f"{stats['gen_rows']:,}")

st.divider()

# ── Page cards ────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ Generation Mix", anchor=False)
    st.markdown(
        "Stacked area chart of fuel-type contributions (Solar, Wind, Gas, Nuclear, …) "
        "by region over time. Filter by region and date range."
    )
    st.subheader("💰 Price Analytics", anchor=False)
    st.markdown(
        "Wholesale price heatmap (hour × day-of-week), price spike table with "
        "rule-based cause classification, and a renewable-share vs price scatter "
        "showing the merit order effect."
    )

with col2:
    st.subheader("🔍 SQL Explorer", anchor=False)
    st.markdown(
        "Write any SQL query against the live DuckDB and render results as a table "
        "or interactive chart. Six pre-built analytical queries included."
    )
    st.subheader("🔮 Demand Forecast", anchor=False)
    st.markdown(
        "Infrastructure for an XGBoost demand forecasting model. "
        "Feature engineering pipeline is ready; model training coming soon."
    )

st.divider()
st.caption(f"Backend: {db.backend_label(con)}  ·  Last refresh: {last}")
