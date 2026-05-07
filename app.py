"""
Energy Grid — SQL Explorer
===========================
Streamlit front-end for the electricity market DuckDB.
Deployable to Streamlit Cloud via MotherDuck (cloud) or local file (dev).

Secrets required for MotherDuck (set in Streamlit Cloud dashboard):
  MOTHERDUCK_TOKEN  — your MotherDuck access token
  MOTHERDUCK_DB     — database name (default: "energy")
"""

import textwrap

import pandas as pd
import plotly.express as px
import streamlit as st

import db

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Energy Grid · SQL Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

# Monospace editor font + tighten default padding
st.markdown(
    """
    <style>
    textarea[aria-label="sql_editor"] {
        font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
    }
    div[data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []   # list of SQL strings, most recent first

if "sql" not in st.session_state:
    st.session_state.sql = textwrap.dedent("""\
        SELECT
            region_id,
            DATE_TRUNC('month', hour) AS month,
            AVG(demand_mwh)           AS avg_demand_mwh,
            MAX(demand_mwh)           AS peak_demand_mwh
        FROM fact_demand
        GROUP BY 1, 2
        ORDER BY 1, 2
        LIMIT 500
    """)

# ── Canned queries ─────────────────────────────────────────────────────────────

EXAMPLES = {
    "Monthly demand by region": textwrap.dedent("""\
        SELECT
            region_id,
            DATE_TRUNC('month', hour) AS month,
            AVG(demand_mwh)           AS avg_demand_mwh,
            MAX(demand_mwh)           AS peak_demand_mwh
        FROM fact_demand
        GROUP BY 1, 2
        ORDER BY 1, 2
        LIMIT 500
    """),
    "Renewable share by region & year": textwrap.dedent("""\
        SELECT
            region_id,
            YEAR(hour)                              AS year,
            SUM(CASE WHEN fuel_id IN ('SUN','WND','WAT') THEN generation_mwh END)
                / NULLIF(SUM(generation_mwh), 0)   AS renewable_share
        FROM fact_generation
        GROUP BY 1, 2
        ORDER BY 1, 2
    """),
    "Hourly price distribution (CISO)": textwrap.dedent("""\
        SELECT
            HOUR(hour)            AS hour_of_day,
            AVG(price_usd_mwh)    AS avg_price,
            PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY price_usd_mwh) AS p10,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY price_usd_mwh) AS p90
        FROM fact_prices
        WHERE region_id = 'CISO'
          AND price_type = 'day_ahead'
        GROUP BY 1
        ORDER BY 1
    """),
    "Price spikes (>3σ rolling)": textwrap.dedent("""\
        WITH stats AS (
            SELECT
                region_id,
                hour,
                price_usd_mwh,
                AVG(price_usd_mwh) OVER w  AS rolling_mean,
                STDDEV(price_usd_mwh) OVER w AS rolling_std
            FROM fact_prices
            WHERE price_type = 'day_ahead'
            WINDOW w AS (
                PARTITION BY region_id
                ORDER BY hour
                ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
            )
        )
        SELECT
            region_id,
            hour,
            price_usd_mwh,
            ROUND(rolling_mean, 2)  AS rolling_mean,
            ROUND((price_usd_mwh - rolling_mean) / NULLIF(rolling_std, 0), 2) AS z_score
        FROM stats
        WHERE price_usd_mwh > rolling_mean + 3 * rolling_std
        ORDER BY z_score DESC
        LIMIT 200
    """),
    "CISO duck curve (March averages)": textwrap.dedent("""\
        SELECT
            YEAR(hour)   AS year,
            HOUR(hour)   AS hour_of_day,
            AVG(d.demand_mwh
                - COALESCE(sun.generation_mwh, 0)
                - COALESCE(wnd.generation_mwh, 0)) AS net_load_mwh
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
    "Generation mix snapshot": textwrap.dedent("""\
        SELECT
            region_id,
            fuel_id,
            ROUND(SUM(generation_mwh) / 1e6, 2) AS total_twh
        FROM fact_generation
        WHERE YEAR(hour) = 2023
        GROUP BY 1, 2
        ORDER BY 1, total_twh DESC
    """),
}

# ── Database connection ────────────────────────────────────────────────────────

con = db.get_connection()

# ── Sidebar: schema browser ───────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ Energy Grid")
    st.caption(db.backend_label(con))
    st.divider()

    # Schema browser
    st.subheader("Schema", anchor=False)
    tables = db.list_tables(con)

    if not tables:
        st.warning(
            "No tables found. Run the ingestion pipeline first:\n\n"
            "```\npython ingestion.py --mode full --regions CISO PJM ERCO\n```"
        )
    else:
        for tbl in tables:
            n = db.row_count(con, tbl)
            label = f"**{tbl}**" + (f"  `{n:,} rows`" if n is not None else "")
            with st.expander(label):
                info = db.table_info(con, tbl)
                for _, row in info.iterrows():
                    col  = row.get("column_name") or row.get("Field", "")
                    typ  = row.get("column_type") or row.get("Type", "")
                    null = row.get("null") or row.get("Null", "YES")
                    dot  = "" if str(null).upper() in ("YES", "TRUE", "1") else " ●"
                    st.markdown(
                        f"`{col}` <span style='color:grey;font-size:0.8em'>"
                        f"{typ}{dot}</span>",
                        unsafe_allow_html=True,
                    )

    st.divider()

    # Example queries
    st.subheader("Example queries", anchor=False)
    for name, sql in EXAMPLES.items():
        if st.button(name, use_container_width=True, key=f"ex_{name}"):
            st.session_state.sql = sql

    st.divider()

    # Query history
    if st.session_state.history:
        st.subheader("History", anchor=False)
        for i, past_sql in enumerate(st.session_state.history[:8]):
            preview = past_sql.strip().splitlines()[0][:48] + "…"
            if st.button(preview, key=f"hist_{i}", use_container_width=True):
                st.session_state.sql = past_sql

# ── Main: SQL editor ──────────────────────────────────────────────────────────

st.header("SQL Explorer", anchor=False, divider="gray")

col_editor, col_btn = st.columns([10, 1], vertical_alignment="bottom")

with col_editor:
    sql_input = st.text_area(
        "sql_editor",
        value=st.session_state.sql,
        height=200,
        label_visibility="collapsed",
        key="sql_textarea",
        placeholder="SELECT …",
    )

with col_btn:
    run_clicked = st.button("▶ Run", type="primary", use_container_width=True)

# Row-limit safety rail
max_rows = st.sidebar.number_input("Row limit", min_value=100, max_value=100_000,
                                   value=5_000, step=500)

# ── Execute ───────────────────────────────────────────────────────────────────

if run_clicked and sql_input.strip():
    # Inject LIMIT if the query has no LIMIT clause (case-insensitive)
    sql_to_run = sql_input
    if "limit" not in sql_input.lower():
        # Wrap as subquery to avoid mangling CTEs
        sql_to_run = f"SELECT * FROM ({sql_input.rstrip('; \n')}) __q LIMIT {max_rows}"

    result = db.run_query(con, sql_to_run)

    # Save to history (dedup)
    clean = sql_input.strip()
    if not st.session_state.history or st.session_state.history[0] != clean:
        st.session_state.history.insert(0, clean)
        st.session_state.history = st.session_state.history[:20]

    st.session_state.last_result = result
    st.session_state.sql = sql_input   # persist editor content

# ── Results ───────────────────────────────────────────────────────────────────

if "last_result" in st.session_state:
    result = st.session_state.last_result

    if result.error:
        st.error(f"**Query error** — {result.error}", icon="🚨")
    else:
        df = result.df

        # Metrics row
        m1, m2, m3, _ = st.columns([1, 1, 1, 4])
        m1.metric("Rows", f"{result.row_count:,}")
        m2.metric("Columns", result.col_count)
        m3.metric("Time", f"{result.elapsed_s * 1000:.0f} ms")

        # Data table
        st.dataframe(df, use_container_width=True, height=380)

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download CSV",
            data=csv,
            file_name="query_result.csv",
            mime="text/csv",
        )

        # Chart panel — always shown when there is at least one numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols and not df.empty:
            with st.expander("📊 Chart", expanded=True):
                all_cols = df.columns.tolist()

                # Smart defaults: prefer datetime → categorical for x-axis
                time_cols = [c for c in all_cols
                             if pd.api.types.is_datetime64_any_dtype(df[c])]
                cat_cols  = [c for c in all_cols if c not in num_cols]
                x_default = (time_cols or cat_cols or all_cols)[0]
                x_default_idx = all_cols.index(x_default)

                c1, c2, c3, c4 = st.columns(4)
                chart_type = c1.selectbox("Chart type", ["line", "bar", "scatter"],
                                          key="chart_type_sel")
                x_col = c2.selectbox("X axis", all_cols, index=x_default_idx,
                                     key="chart_x")
                y_col = c3.selectbox("Y axis", num_cols, index=0, key="chart_y")
                color_col = c4.selectbox(
                    "Colour by",
                    ["(none)"] + [c for c in all_cols if c not in (x_col, y_col)],
                    key="chart_color",
                )
                color_arg = None if color_col == "(none)" else color_col

                kwargs = dict(x=x_col, y=y_col, color=color_arg,
                              template="plotly_white")
                if chart_type == "line":
                    fig = px.line(df, **kwargs)
                elif chart_type == "bar":
                    fig = px.bar(df, **kwargs)
                else:
                    fig = px.scatter(df, **kwargs)

                fig.update_layout(margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
