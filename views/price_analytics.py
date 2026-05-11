import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import db
from constants import PRICE_REGIONS, REGION_COLORS, DOW_LABELS

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_heatmap(region: str, year: int) -> pd.DataFrame | None:
    con = db.get_connection()
    result = db.run_query(con, f"""
        SELECT
            ISODOW(hour)  AS dow,
            HOUR(hour)    AS hour_of_day,
            ROUND(AVG(price_usd_mwh), 2) AS avg_price
        FROM fact_prices
        WHERE region_id  = '{region}'
          AND price_type = 'day_ahead'
          AND YEAR(hour) = {year}
          AND price_usd_mwh BETWEEN -100 AND 1000
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)
    return result.df if not result.error else None


@st.cache_data(ttl=300)
def load_spikes(region: str, year: int) -> pd.DataFrame | None:
    con = db.get_connection()
    result = db.run_query(con, f"""
        WITH stats AS (
            SELECT
                hour, price_usd_mwh,
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
            ROUND(price_usd_mwh, 2) AS price_usd_mwh,
            ROUND((price_usd_mwh - rolling_mean) / NULLIF(rolling_std, 0), 2) AS z_score,
            CASE
                WHEN HOUR(hour) BETWEEN 14 AND 21 AND MONTH(hour) IN (6,7,8)
                    THEN 'Summer Peak Demand'
                WHEN HOUR(hour) BETWEEN 17 AND 22 AND MONTH(hour) IN (12,1,2)
                    THEN 'Winter Heating Demand'
                WHEN HOUR(hour) BETWEEN 0 AND 5
                    THEN 'Off-Peak Supply Shortage'
                ELSE 'Transmission / Other'
            END AS cause
        FROM stats
        WHERE price_usd_mwh > rolling_mean + 3 * rolling_std
        ORDER BY z_score DESC
        LIMIT 200
    """)
    return result.df if not result.error else None


@st.cache_data(ttl=300)
def load_merit_order(start_year: int, end_year: int) -> pd.DataFrame | None:
    con = db.get_connection()
    result = db.run_query(con, f"""
        SELECT
            g.region_id,
            YEAR(g.hour)                       AS year,
            DATE_TRUNC('week', g.hour)::DATE   AS week,
            ROUND(
                SUM(CASE WHEN g.fuel_id IN ('SUN','WND','WAT') THEN g.generation_mwh ELSE 0 END)
                / NULLIF(SUM(g.generation_mwh), 0), 3
            ) AS ren_share,
            ROUND(AVG(p.price_usd_mwh), 2) AS avg_price,
            CASE MONTH(g.hour)
                WHEN 12 THEN 'Winter' WHEN 1 THEN 'Winter' WHEN 2 THEN 'Winter'
                WHEN 3  THEN 'Spring' WHEN 4 THEN 'Spring' WHEN 5 THEN 'Spring'
                WHEN 6  THEN 'Summer' WHEN 7 THEN 'Summer' WHEN 8 THEN 'Summer'
                ELSE 'Fall'
            END AS season
        FROM fact_generation g
        JOIN fact_prices p
          ON p.hour = g.hour AND p.region_id = g.region_id AND p.price_type = 'day_ahead'
        WHERE YEAR(g.hour) BETWEEN {start_year} AND {end_year}
          AND g.region_id IN ('CISO', 'PJM', 'NYIS', 'ISNE')
          AND p.price_usd_mwh BETWEEN -50 AND 500
        GROUP BY 1, 2, 3, 6
        HAVING ren_share IS NOT NULL
        ORDER BY 1, 3
    """)
    return result.df if not result.error else None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters", anchor=False)
    region     = st.selectbox("Region", PRICE_REGIONS, index=0)
    year       = st.selectbox("Year", list(range(2024, 2013, -1)), index=0)
    year_range = st.slider("Merit order year range", 2015, 2024, (2020, 2024))
    start_year, end_year = year_range

# ── Page ──────────────────────────────────────────────────────────────────────

st.header("Price Analytics", anchor=False, divider="gray")

tab_heat, tab_spikes, tab_merit = st.tabs(
    ["🌡️ Price Heatmap", "⚡ Price Spikes", "🌿 Merit Order Effect"]
)

# ── Tab 1: Heatmap ────────────────────────────────────────────────────────────

with tab_heat:
    st.subheader(f"Avg wholesale price — {region} {year}", anchor=False)
    df_heat = load_heatmap(region, year)

    if df_heat is None or df_heat.empty:
        st.warning("No price data for the selected filters.")
    else:
        df_heat["dow_name"] = df_heat["dow"].map(DOW_LABELS)
        pivot = (
            df_heat.pivot(index="hour_of_day", columns="dow_name", values="avg_price")
            .reindex(columns=list(DOW_LABELS.values()))
        )
        fig = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            labels=dict(x="Day of Week", y="Hour of Day", color="$/MWh"),
            template="plotly_dark",
        )
        fig.update_layout(height=440, margin=dict(t=20, b=20))
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, width="stretch")
        st.caption("Green = low price  ·  Red = high price  ·  Hover for exact value")

# ── Tab 2: Price spikes ───────────────────────────────────────────────────────

with tab_spikes:
    st.subheader(f"Price spikes (>3σ) — {region} {year}", anchor=False)
    df_spikes = load_spikes(region, year)

    if df_spikes is None or df_spikes.empty:
        st.info("No spikes detected for the selected filters.")
    else:
        # Cause distribution
        cause_counts = df_spikes["cause"].value_counts().reset_index()
        cause_counts.columns = ["cause", "count"]
        col_bar, col_table = st.columns([1, 2])

        with col_bar:
            fig_bar = px.bar(
                cause_counts, x="count", y="cause", orientation="h",
                color="cause", template="plotly_dark",
                labels={"count": "# Spikes", "cause": ""},
            )
            fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_bar, width="stretch")

        with col_table:
            st.dataframe(
                df_spikes[["hour", "price_usd_mwh", "z_score", "cause"]].rename(columns={
                    "hour": "Hour", "price_usd_mwh": "Price ($/MWh)",
                    "z_score": "Z-score", "cause": "Classified Cause",
                }),
                hide_index=True,
                width="stretch",
                height=320,
            )

# ── Tab 3: Merit order scatter ────────────────────────────────────────────────

with tab_merit:
    st.subheader(
        f"Renewable share vs wholesale price — {start_year}–{end_year}", anchor=False
    )
    df_merit = load_merit_order(start_year, end_year)

    if df_merit is None or df_merit.empty:
        st.warning("No data for the selected year range.")
    else:
        color_by = st.radio(
            "Colour points by", ["Region", "Year"], horizontal=True, key="merit_color"
        )
        color_col = "region_id" if color_by == "Region" else "year"
        color_map = REGION_COLORS if color_by == "Region" else None

        fig_scatter = px.scatter(
            df_merit,
            x="ren_share",
            y="avg_price",
            color=color_col,
            color_discrete_map=color_map,
            facet_col="season",
            facet_col_wrap=2,
            trendline="ols",
            labels={
                "ren_share":  "Renewable Share",
                "avg_price":  "Avg Price ($/MWh)",
                "region_id":  "Region",
                "year":       "Year",
            },
            template="plotly_dark",
            opacity=0.6,
        )
        fig_scatter.update_layout(height=520, margin=dict(t=40, b=20))
        st.plotly_chart(fig_scatter, width="stretch")
        st.caption(
            "Each point = one week average.  "
            "Downward-sloping OLS line confirms the merit order effect: "
            "higher renewable share compresses day-ahead prices."
        )
