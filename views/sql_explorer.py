import textwrap

import pandas as pd
import plotly.express as px
import streamlit as st

import db
from constants import PRICE_TYPE_DAY_AHEAD

# ── Pre-built analytical queries ──────────────────────────────────────────────

PREBUILT = {
    "Top 10 highest price hours this year": textwrap.dedent(f"""\
        SELECT
            hour,
            region_id,
            ROUND(price_usd_mwh, 2)  AS price_usd_mwh
        FROM fact_prices
        WHERE price_type = '{PRICE_TYPE_DAY_AHEAD}'
          AND YEAR(hour) = YEAR(CURRENT_DATE)
        ORDER BY price_usd_mwh DESC
        LIMIT 10
    """),
    "Renewable penetration by region (annual)": textwrap.dedent("""\
        SELECT
            YEAR(hour)    AS year,
            region_id,
            ROUND(100.0 * SUM(CASE WHEN fuel_id IN ('SUN','WND','WAT')
                                   THEN generation_mwh END)
                  / NULLIF(SUM(generation_mwh), 0), 1)  AS renewable_pct
        FROM fact_generation
        GROUP BY 1, 2
        ORDER BY 2, 1
    """),
    "Peak demand hours by region (top 1%)": textwrap.dedent("""\
        SELECT region_id, hour, ROUND(demand_mwh, 0) AS demand_mwh
        FROM fact_demand
        QUALIFY PERCENT_RANK() OVER (
            PARTITION BY region_id ORDER BY demand_mwh
        ) >= 0.99
        ORDER BY region_id, demand_mwh DESC
        LIMIT 100
    """),
    "Price spike summary by region": textwrap.dedent(f"""\
        WITH stats AS (
            SELECT
                region_id, hour, price_usd_mwh,
                AVG(price_usd_mwh) OVER w  AS rolling_mean,
                STDDEV(price_usd_mwh) OVER w AS rolling_std
            FROM fact_prices
            WHERE price_type = '{PRICE_TYPE_DAY_AHEAD}'
            WINDOW w AS (
                PARTITION BY region_id
                ORDER BY hour
                ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
            )
        )
        SELECT
            region_id,
            COUNT(*)                                       AS spike_count,
            ROUND(AVG(price_usd_mwh), 2)                  AS avg_spike_price,
            ROUND(MAX(price_usd_mwh), 2)                   AS max_price,
            ROUND(AVG((price_usd_mwh - rolling_mean)
                      / NULLIF(rolling_std, 0)), 2)        AS avg_z_score
        FROM stats
        WHERE price_usd_mwh > rolling_mean + 3 * rolling_std
        GROUP BY region_id
        ORDER BY spike_count DESC
    """),
    "Price vs demand correlation by region": textwrap.dedent(f"""\
        SELECT
            d.region_id,
            YEAR(d.hour)                                   AS year,
            ROUND(CORR(d.demand_mwh, p.price_usd_mwh), 3) AS corr_demand_price,
            COUNT(*)                                        AS n_hours
        FROM fact_demand d
        JOIN fact_prices p
          ON p.hour = d.hour
         AND p.region_id = d.region_id
         AND p.price_type = '{PRICE_TYPE_DAY_AHEAD}'
        GROUP BY 1, 2
        ORDER BY 1, 2
    """),
    "CISO duck curve (March net load by year)": textwrap.dedent("""\
        SELECT
            YEAR(d.hour)   AS year,
            HOUR(d.hour)   AS hour_of_day,
            ROUND(AVG(
                d.demand_mwh
                - COALESCE(sun.generation_mwh, 0)
                - COALESCE(wnd.generation_mwh, 0)
            ), 0)          AS net_load_mwh
        FROM fact_demand d
        LEFT JOIN fact_generation sun
               ON sun.hour = d.hour AND sun.region_id = d.region_id AND sun.fuel_id = 'SUN'
        LEFT JOIN fact_generation wnd
               ON wnd.hour = d.hour AND wnd.region_id = d.region_id AND wnd.fuel_id = 'WND'
        WHERE d.region_id = 'CISO'
          AND MONTH(d.hour) = 3
        GROUP BY 1, 2
        ORDER BY 1, 2
    """),
}

# ── Session state ─────────────────────────────────────────────────────────────

if "sql" not in st.session_state:
    st.session_state.sql = list(PREBUILT.values())[0]

if "history" not in st.session_state:
    st.session_state.history = []

# ── CSS: monospace editor ─────────────────────────────────────────────────────

st.markdown(
    """<style>
    textarea { font-family:"JetBrains Mono","Fira Code",monospace !important;
               font-size:13px !important; line-height:1.6 !important; }
    div[data-testid="stMetricValue"] { font-size:1.3rem !important; }
    </style>""",
    unsafe_allow_html=True,
)

con = db.get_connection()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Schema", anchor=False)
    tables = db.list_tables(con)
    if not tables:
        st.warning("No tables found — run ingestion first.")
    else:
        for tbl in tables:
            n = db.row_count(con, tbl)
            label = f"**{tbl}**" + (f"  `{n:,}`" if n else "")
            with st.expander(label):
                info = db.table_info(con, tbl)
                for _, row in info.iterrows():
                    col = row.get("column_name") or row.get("Field", "")
                    typ = row.get("column_type") or row.get("Type", "")
                    st.markdown(
                        f"`{col}` <span style='color:grey;font-size:0.8em'>{typ}</span>",
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.header("History", anchor=False)
    for i, past in enumerate(st.session_state.history[:8]):
        preview = past.strip().splitlines()[0][:50] + "…"
        if st.button(preview, key=f"hist_{i}", width="stretch"):
            st.session_state.sql = past

# ── Main ──────────────────────────────────────────────────────────────────────

st.header("SQL Explorer", anchor=False, divider="gray")

# Pre-built query tiles
st.subheader("Pre-built queries", anchor=False)
cols = st.columns(3)
for i, (name, sql) in enumerate(PREBUILT.items()):
    if cols[i % 3].button(name, key=f"pb_{i}", width="stretch"):
        st.session_state.sql = sql

st.divider()

# Editor
max_rows  = st.sidebar.number_input("Row limit", 100, 100_000, 5_000, 500)
col_ed, col_btn = st.columns([10, 1], vertical_alignment="bottom")

with col_ed:
    sql_input = st.text_area(
        "sql_editor",
        value=st.session_state.sql,
        height=200,
        label_visibility="collapsed",
        key="sql_textarea",
    )
with col_btn:
    run = st.button("▶ Run", type="primary", width="stretch")

# ── Execute & render ──────────────────────────────────────────────────────────

if run and sql_input.strip():
    sql_to_run = sql_input
    if "limit" not in sql_input.lower():
        sql_to_run = f"SELECT * FROM ({sql_input.rstrip('; \n')}) __q LIMIT {max_rows}"

    result = db.run_query(con, sql_to_run)

    clean = sql_input.strip()
    if not st.session_state.history or st.session_state.history[0] != clean:
        st.session_state.history.insert(0, clean)
        st.session_state.history = st.session_state.history[:20]

    st.session_state.last_result = result
    st.session_state.sql = sql_input

if "last_result" in st.session_state:
    result = st.session_state.last_result

    if result.error:
        st.error(f"**Query error** — {result.error}")
    else:
        df = result.df
        m1, m2, m3, _ = st.columns([1, 1, 1, 4])
        m1.metric("Rows",    f"{result.row_count:,}")
        m2.metric("Columns", result.col_count)
        m3.metric("Time",    f"{result.elapsed_s * 1000:.0f} ms")

        st.dataframe(df, width="stretch", height=360)
        st.download_button("Download CSV", df.to_csv(index=False).encode(),
                           "query_result.csv", "text/csv")

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols and not df.empty:
            with st.expander("Chart", expanded=True):
                all_cols  = df.columns.tolist()
                time_cols = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
                cat_cols  = [c for c in all_cols if c not in num_cols]
                x_default = (time_cols or cat_cols or all_cols)[0]

                c1, c2, c3, c4 = st.columns(4)
                chart_type = c1.selectbox("Type",   ["line","bar","scatter"], key="ctype")
                x_col      = c2.selectbox("X axis", all_cols,
                                          index=all_cols.index(x_default), key="cx")
                y_col      = c3.selectbox("Y axis", num_cols, key="cy")
                color_opts = ["(none)"] + [c for c in all_cols if c not in (x_col, y_col)]
                color_def  = next((c for c in cat_cols if c not in (x_col, y_col)), "(none)")
                color_col  = c4.selectbox("Colour by", color_opts,
                                          index=color_opts.index(color_def), key="cc")
                color_arg  = None if color_col == "(none)" else color_col

                kw = dict(x=x_col, y=y_col, color=color_arg, template="plotly_dark")
                fig = (px.line if chart_type == "line" else
                       px.bar  if chart_type == "bar"  else px.scatter)(df, **kw)
                fig.update_layout(margin=dict(t=20, b=20))
                st.plotly_chart(fig, width="stretch")
