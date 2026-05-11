"""
EDA — US Electricity Market
Seven interactive analyses, each driven by live DuckDB queries.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from constants import REGIONS, PRICE_REGIONS, REGION_COLORS, FUEL_COLORS, FUEL_LABELS
import eda

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters", anchor=False)
    region = st.selectbox("Region", REGIONS, index=0)
    year   = st.selectbox("Year",   list(range(2024, 2013, -1)), index=0)

st.header("Exploratory Data Analysis", anchor=False, divider="gray")

(
    tab_stl, tab_mstl, tab_corr,
    tab_peak, tab_duck, tab_spike, tab_heatmap,
) = st.tabs([
    "📉 STL",
    "🔀 MSTL",
    "🌿 Ren vs Price",
    "🔥 Peak Demand",
    "🦆 Duck Curve",
    "⚡ Price Spikes",
    "🎯 Forecast MAPE",
])

_DARK = "plotly_dark"

# ══════════════════════════════════════════════════════════════════════════════
# 1 · STL Decomposition
# ══════════════════════════════════════════════════════════════════════════════

with tab_stl:
    st.subheader(f"STL Decomposition — {region} {year}", anchor=False)
    st.caption(
        "Seasonal-Trend decomposition using Loess (period = 24 h).  "
        "Robust fitting down-weights residual outliers."
    )

    @st.cache_data(ttl=300, show_spinner=False)
    def _stl(region, year):
        df = eda.load_demand_series(region, year)
        return eda.stl_decompose(df) if df is not None else None

    with st.spinner("Running STL…"):
        stl_df = _stl(region, year)

    if stl_df is None:
        st.warning("No demand data for the selected region / year.")
    else:
        panels = [
            ("demand_mwh", "Observed (MWh)",  "#4fc3f7"),
            ("trend",      "Trend (MWh)",      "#ff7f0e"),
            ("seasonal",   "Daily Seasonal",   "#2ca02c"),
            ("resid",      "Residual",         "#d62728"),
        ]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                             vertical_spacing=0.04,
                             subplot_titles=[p[1] for p in panels])
        for i, (col, title, color) in enumerate(panels, 1):
            mode = "markers" if col == "resid" else "lines"
            fig.add_trace(
                go.Scatter(x=stl_df["hour"], y=stl_df[col],
                           mode=mode, name=title,
                           line=dict(color=color, width=1) if mode == "lines"
                                else None,
                           marker=dict(color=color, size=2, opacity=0.5)
                                if mode == "markers" else None),
                row=i, col=1,
            )
        fig.update_layout(template=_DARK, height=700,
                          margin=dict(t=30, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

        with st.expander("Residual distribution"):
            fig_hist = px.histogram(
                stl_df, x="resid", nbins=80,
                labels={"resid": "Residual (MWh)"},
                template=_DARK, color_discrete_sequence=["#d62728"],
            )
            fig_hist.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig_hist, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# 2 · MSTL Decomposition
# ══════════════════════════════════════════════════════════════════════════════

with tab_mstl:
    st.subheader(f"MSTL Decomposition — {region} {year}", anchor=False)
    st.caption(
        "Multiple Seasonal-Trend decomposition — extracts a **daily (24 h)** "
        "and **weekly (168 h)** seasonal component simultaneously."
    )

    @st.cache_data(ttl=300, show_spinner=False)
    def _mstl(region, year):
        df = eda.load_demand_series(region, year)
        return eda.mstl_decompose(df) if df is not None else None

    with st.spinner("Running MSTL…"):
        mstl_df = _mstl(region, year)

    if mstl_df is None:
        st.warning("No demand data for the selected region / year.")
    else:
        panels = [
            ("demand_mwh",       "Observed (MWh)",      "#4fc3f7"),
            ("trend",            "Trend (MWh)",          "#ff7f0e"),
            ("seasonal_daily",   "Daily Seasonal",       "#2ca02c"),
            ("seasonal_weekly",  "Weekly Seasonal",      "#9467bd"),
            ("resid",            "Residual",             "#d62728"),
        ]
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                             vertical_spacing=0.03,
                             subplot_titles=[p[1] for p in panels])
        for i, (col, title, color) in enumerate(panels, 1):
            mode = "markers" if col == "resid" else "lines"
            fig.add_trace(
                go.Scatter(x=mstl_df["hour"], y=mstl_df[col],
                           mode=mode, name=title,
                           line=dict(color=color, width=1) if mode == "lines"
                                else None,
                           marker=dict(color=color, size=2, opacity=0.4)
                                if mode == "markers" else None),
                row=i, col=1,
            )
        fig.update_layout(template=_DARK, height=820,
                          margin=dict(t=30, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

        # Seasonal amplitude comparison
        with st.expander("Seasonal amplitude by month"):
            mstl_df["month"] = mstl_df["hour"].dt.month
            amp = (
                mstl_df.groupby("month")[["seasonal_daily", "seasonal_weekly"]]
                .std().round(1).reset_index()
                .rename(columns={"seasonal_daily": "Daily σ (MWh)",
                                  "seasonal_weekly": "Weekly σ (MWh)"})
            )
            fig_amp = px.bar(
                amp.melt(id_vars="month", var_name="Component", value_name="Std Dev (MWh)"),
                x="month", y="Std Dev (MWh)", color="Component", barmode="group",
                template=_DARK,
                labels={"month": "Month"},
            )
            fig_amp.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig_amp, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# 3 · Renewable vs Price Correlation
# ══════════════════════════════════════════════════════════════════════════════

with tab_corr:
    eff_region = region if region in PRICE_REGIONS else PRICE_REGIONS[0]
    if region not in PRICE_REGIONS:
        st.info(
            f"Price data not available for {region}. Showing **{eff_region}** instead.",
            icon="ℹ️",
        )

    st.subheader(f"Renewable Share vs Wholesale Price — {eff_region} {year}",
                  anchor=False)
    st.caption(
        "Merit order effect: higher renewable penetration displaces gas peakers, "
        "compressing day-ahead prices."
    )

    @st.cache_data(ttl=300, show_spinner=False)
    def _ren_price(region, year):
        return eda.load_ren_price(region, year)

    with st.spinner("Loading…"):
        rp_df = _ren_price(eff_region, year)

    if rp_df is None:
        st.warning("No data for the selected region / year.")
    else:
        col_scatter, col_corr = st.columns([2, 1])

        with col_scatter:
            fig_sc = px.scatter(
                rp_df.sample(min(5000, len(rp_df)), random_state=42),
                x="ren_share", y="price_usd_mwh", color="season",
                trendline="ols",
                opacity=0.45,
                labels={"ren_share": "Renewable Share",
                         "price_usd_mwh": "Day-Ahead Price ($/MWh)",
                         "season": "Season"},
                template=_DARK,
            )
            fig_sc.update_layout(height=380, margin=dict(t=10, b=10))
            st.plotly_chart(fig_sc, width="stretch")

        with col_corr:
            corr_df = eda.ren_price_hourly_corr(rp_df)
            overall = rp_df[["ren_share", "price_usd_mwh"]].corr().iloc[0, 1]
            st.metric("Overall Pearson r", f"{overall:.3f}")
            st.caption("Correlation by hour-of-day:")
            fig_bar = px.bar(
                corr_df, x="correlation", y="hour_of_day",
                orientation="h",
                color="correlation",
                color_continuous_scale="RdBu",
                range_color=[-1, 1],
                labels={"hour_of_day": "Hour", "correlation": "r"},
                template=_DARK,
            )
            fig_bar.update_layout(height=380, margin=dict(t=10, b=10, l=10),
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_bar, width="stretch")

        # Binned heatmap: ren_share bucket × hour_of_day → avg price
        with st.expander("Price heatmap: renewable bucket × hour"):
            rp_df["ren_bucket"] = pd.cut(
                rp_df["ren_share"],
                bins=[0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01],
                labels=["0–10%","10–20%","20–30%","30–40%",
                         "40–50%","50–60%","60–70%","70–80%","80%+"],
            )
            pivot = (
                rp_df.groupby(["ren_bucket", "hour_of_day"])["price_usd_mwh"]
                .mean().round(2).unstack("hour_of_day")
            )
            fig_hm = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn_r",
                labels=dict(x="Hour of Day", y="Renewable Share Bucket",
                             color="$/MWh"),
                aspect="auto",
                template=_DARK,
            )
            fig_hm.update_layout(height=380, margin=dict(t=10, b=10))
            st.plotly_chart(fig_hm, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# 4 · Peak Demand Profiling
# ══════════════════════════════════════════════════════════════════════════════

with tab_peak:
    top_pct = st.slider("Top % of demand hours", 0.5, 5.0, 1.0, 0.5,
                          key="peak_pct", format="%.1f%%") / 100

    st.subheader(f"Peak Demand Profile — {region} {year} (top {top_pct*100:.1f}%)",
                  anchor=False)

    @st.cache_data(ttl=300, show_spinner=False)
    def _peak(region, year, top_pct):
        return eda.load_peak_profile(region, year, top_pct)

    with st.spinner("Loading peak hours…"):
        prof = _peak(region, year, top_pct)

    if prof is None or prof["peaks"] is None:
        st.warning("No data for the selected region / year.")
    else:
        peaks = prof["peaks"]
        st.metric("Demand threshold", f"{prof['cutoff_mwh']:,.0f} MWh")

        col_hod, col_dow = st.columns(2)

        with col_hod:
            hod = peaks["hour_of_day"].value_counts().sort_index().reset_index()
            hod.columns = ["Hour of Day", "Count"]
            fig_hod = px.bar(hod, x="Hour of Day", y="Count",
                              template=_DARK, color="Count",
                              color_continuous_scale="Reds")
            fig_hod.update_layout(height=280, margin=dict(t=10, b=10),
                                   title="Hour of Day",
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_hod, width="stretch")

        with col_dow:
            dow_labels = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
            dow = peaks["day_of_week"].map(dow_labels).value_counts()
            dow = dow.reindex(list(dow_labels.values()), fill_value=0).reset_index()
            dow.columns = ["Day", "Count"]
            fig_dow = px.bar(dow, x="Day", y="Count",
                              template=_DARK, color="Count",
                              color_continuous_scale="Oranges")
            fig_dow.update_layout(height=280, margin=dict(t=10, b=10),
                                   title="Day of Week",
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_dow, width="stretch")

        # Season pie
        col_season, col_fuel = st.columns(2)

        with col_season:
            season_ct = peaks["season"].value_counts().reset_index()
            season_ct.columns = ["Season", "Count"]
            fig_pie = px.pie(season_ct, values="Count", names="Season",
                              template=_DARK, hole=0.4,
                              color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pie.update_layout(height=300, margin=dict(t=10, b=10),
                                   title="Season mix")
            st.plotly_chart(fig_pie, width="stretch")

        with col_fuel:
            if prof["fuel_peak"] is not None and prof["fuel_base"] is not None:
                fp = prof["fuel_peak"].copy()
                fb = prof["fuel_base"].copy()
                fp["period"] = "Peak"
                fb["period"] = "Baseline"
                fuel_df = pd.concat([fp, fb])
                fuel_df["fuel_label"] = fuel_df["fuel_id"].map(FUEL_LABELS).fillna(fuel_df["fuel_id"])
                fuel_df["color"]      = fuel_df["fuel_id"].map(FUEL_COLORS)
                fig_fuel = px.bar(
                    fuel_df, x="fuel_label", y="avg_gen_mwh", color="period",
                    barmode="group", template=_DARK,
                    labels={"avg_gen_mwh": "Avg Gen (MWh)", "fuel_label": "Fuel",
                             "period": ""},
                )
                fig_fuel.update_layout(height=300, margin=dict(t=10, b=10),
                                        title="Fuel mix: peak vs baseline")
                st.plotly_chart(fig_fuel, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# 5 · Duck Curve
# ══════════════════════════════════════════════════════════════════════════════

with tab_duck:
    duck_region = st.selectbox("Region", REGIONS,
                                index=REGIONS.index("CISO"), key="duck_region")
    month_sel = st.selectbox(
        "Month (select to filter; '0 = all year' shows March by default)",
        [0] + list(range(1, 13)),
        index=3,
        format_func=lambda m: "All months" if m == 0 else
            ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
        key="duck_month",
    )

    st.subheader(f"Duck Curve — {duck_region}", anchor=False)
    st.caption(
        "Net load = Gross demand − Solar − Wind.  "
        "The deepening mid-day valley shows solar's growing displacement of conventional generation."
    )

    @st.cache_data(ttl=300, show_spinner=False)
    def _duck(region):
        return eda.load_duck_curve(region)

    with st.spinner("Loading…"):
        duck_df = _duck(duck_region)

    if duck_df is None:
        st.warning("No data available.")
    else:
        duck_plot = duck_df.copy()
        if month_sel != 0:
            duck_plot = duck_plot[duck_plot["month"] == month_sel]
        duck_plot = (
            duck_plot.groupby(["year", "hour_of_day"])["net_load_mwh"]
            .mean().round(0).reset_index()
        )

        year_range = sorted(duck_plot["year"].unique())
        colors = px.colors.sample_colorscale(
            "Viridis", [i / max(len(year_range) - 1, 1) for i in range(len(year_range))]
        )

        fig_duck = go.Figure()
        for yr, color in zip(year_range, colors):
            sub = duck_plot[duck_plot["year"] == yr]
            fig_duck.add_trace(go.Scatter(
                x=sub["hour_of_day"], y=sub["net_load_mwh"],
                name=str(yr), mode="lines",
                line=dict(color=color, width=2),
            ))
        fig_duck.update_layout(
            template=_DARK, height=420,
            margin=dict(t=10, b=10),
            xaxis=dict(title="Hour of Day", tickmode="linear", dtick=2),
            yaxis_title="Net Load (MWh)",
            legend=dict(title="Year", orientation="v"),
        )
        st.plotly_chart(fig_duck, width="stretch")

        # Ramp rate panel
        with st.expander("Evening ramp steepness (16:00 → 20:00)"):
            ramp = (
                duck_plot[duck_plot["hour_of_day"].between(16, 20)]
                .groupby("year").apply(
                    lambda g: g.set_index("hour_of_day")["net_load_mwh"].diff().dropna().sum()
                )
                .reset_index()
                .rename(columns={0: "ramp_mwh"})
            )
            fig_ramp = px.bar(ramp, x="year", y="ramp_mwh",
                               template=_DARK,
                               labels={"ramp_mwh": "Net Load Ramp (MWh)", "year": "Year"},
                               color="ramp_mwh", color_continuous_scale="Reds")
            fig_ramp.update_layout(height=260, margin=dict(t=10, b=10),
                                    coloraxis_showscale=False)
            st.plotly_chart(fig_ramp, width="stretch")
            st.caption("Larger positive value = steeper evening ramp pressure on dispatchable generators.")


# ══════════════════════════════════════════════════════════════════════════════
# 6 · Price Spike Characterisation
# ══════════════════════════════════════════════════════════════════════════════

with tab_spike:
    spike_region = region if region in PRICE_REGIONS else PRICE_REGIONS[0]
    if region not in PRICE_REGIONS:
        st.info(f"Price data not available for {region}. Showing **{spike_region}**.", icon="ℹ️")

    st.subheader(f"Price Spike Characterisation — {spike_region} {year}", anchor=False)
    st.caption("Spikes defined as hours where price > rolling 168h mean + 3σ.")

    @st.cache_data(ttl=300, show_spinner=False)
    def _spikes(region, year):
        return eda.load_spikes_full(region, year)

    with st.spinner("Loading spikes…"):
        sp_df = _spikes(spike_region, year)

    if sp_df is None or sp_df.empty:
        st.info("No spikes detected for the selected filters.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total spike hours",   f"{len(sp_df):,}")
        m2.metric("Max price ($/MWh)",   f"{sp_df['price_usd_mwh'].max():,.0f}")
        m3.metric("Avg z-score",          f"{sp_df['z_score'].mean():.1f}σ")
        m4.metric("Months affected",      f"{sp_df['month'].nunique()}")

        col_mag, col_freq = st.columns(2)

        with col_mag:
            fig_mag = px.histogram(
                sp_df, x="price_usd_mwh", nbins=40,
                color="season",
                labels={"price_usd_mwh": "Price ($/MWh)"},
                template=_DARK,
                barmode="overlay", opacity=0.7,
            )
            fig_mag.update_layout(height=300, margin=dict(t=10, b=10),
                                   title="Spike magnitude distribution")
            st.plotly_chart(fig_mag, width="stretch")

        with col_freq:
            monthly = eda.spike_monthly_counts(sp_df)
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"]
            monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m-1])
            fig_freq = px.bar(monthly, x="month_name", y="spike_count",
                               color="avg_price", color_continuous_scale="YlOrRd",
                               labels={"spike_count": "# Spikes",
                                        "month_name": "Month",
                                        "avg_price": "Avg $/MWh"},
                               template=_DARK)
            fig_freq.update_layout(height=300, margin=dict(t=10, b=10),
                                    title="Spike frequency by month")
            st.plotly_chart(fig_freq, width="stretch")

        # Hour-of-day distribution
        hod_sp = sp_df["hour_of_day"].value_counts().sort_index().reset_index()
        hod_sp.columns = ["Hour of Day", "Spike Count"]
        fig_hod = px.bar(hod_sp, x="Hour of Day", y="Spike Count",
                          template=_DARK, color="Spike Count",
                          color_continuous_scale="YlOrRd")
        fig_hod.update_layout(height=240, margin=dict(t=10, b=10),
                               title="Spikes by hour-of-day",
                               coloraxis_showscale=False)
        st.plotly_chart(fig_hod, width="stretch")

        # Duration analysis
        with st.expander("Spike duration analysis"):
            runs = eda.spike_duration_stats(sp_df)
            col_dur, col_tbl = st.columns([1, 2])
            with col_dur:
                fig_dur = px.histogram(runs, x="n_hours", nbins=20,
                                        labels={"n_hours": "Duration (hours)"},
                                        template=_DARK,
                                        color_discrete_sequence=["#ff7f0e"])
                fig_dur.update_layout(height=260, margin=dict(t=10, b=10),
                                       title="Consecutive spike run length")
                st.plotly_chart(fig_dur, width="stretch")
            with col_tbl:
                st.dataframe(
                    runs.sort_values("max_price", ascending=False)
                    .head(20)
                    .rename(columns={"n_hours": "Hours", "max_price": "Max $/MWh",
                                      "max_z": "Max Z", "season": "Season"}),
                    hide_index=True, height=260, width="stretch",
                )


# ══════════════════════════════════════════════════════════════════════════════
# 7 · Forecast Accuracy Heatmap
# ══════════════════════════════════════════════════════════════════════════════

with tab_heatmap:
    st.subheader(f"XGBoost Forecast MAPE by Hour-of-Day — {region}", anchor=False)
    st.caption(
        "Expanding-window CV (8 folds, 24h horizon).  "
        "Which hours are hardest to forecast?"
    )

    mape_train_years = st.slider("Training window (years)", 1, 3, 2, key="mape_ty")
    run_mape = st.button("▶ Run Forecast CV", type="primary", key="run_mape_btn")

    if run_mape:
        with st.spinner("Training XGBoost across CV folds…"):
            mape_df = eda.forecast_mape_heatmap(
                region, train_years=mape_train_years, n_folds=8
            )
        st.session_state["mape_result"] = (region, mape_df)

    cached_mape = st.session_state.get("mape_result")
    if cached_mape and cached_mape[0] == region:
        mape_df = cached_mape[1]
        if mape_df is None:
            st.warning("No data for this region.")
        else:
            col_bar, col_tbl = st.columns([2, 1])

            with col_bar:
                fig_mape = go.Figure()
                fig_mape.add_trace(go.Bar(
                    x=mape_df["hour_of_day"],
                    y=mape_df["MAPE_mean"],
                    error_y=dict(type="data", array=mape_df["MAPE_std"].tolist(),
                                  visible=True, color="#888"),
                    marker_color=mape_df["MAPE_mean"],
                    marker_colorscale="YlOrRd",
                    showscale=False,
                    name="Mean MAPE",
                ))
                fig_mape.update_layout(
                    template=_DARK, height=360,
                    margin=dict(t=10, b=10),
                    xaxis=dict(title="Hour of Day", tickmode="linear", dtick=2),
                    yaxis_title="MAPE (%)",
                )
                st.plotly_chart(fig_mape, width="stretch")

            with col_tbl:
                st.dataframe(
                    mape_df.rename(columns={
                        "hour_of_day": "Hour",
                        "MAPE_mean":   "Mean MAPE (%)",
                        "MAPE_median": "Median MAPE (%)",
                        "MAPE_std":    "Std Dev (%)",
                    }),
                    hide_index=True, height=360, width="stretch",
                )

            # Polar / clock chart
            with st.expander("Clock chart (24h polar)"):
                theta = mape_df["hour_of_day"] * 15  # 360 / 24
                fig_polar = go.Figure(go.Barpolar(
                    r=mape_df["MAPE_mean"],
                    theta=theta,
                    width=[15] * len(mape_df),
                    marker_color=mape_df["MAPE_mean"],
                    marker_colorscale="YlOrRd",
                    showscale=False,
                ))
                fig_polar.update_layout(
                    template=_DARK, height=380,
                    margin=dict(t=10, b=10),
                    polar=dict(
                        angularaxis=dict(
                            tickmode="array",
                            tickvals=list(range(0, 360, 30)),
                            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
                            direction="clockwise",
                            rotation=90,
                        ),
                        radialaxis=dict(title="MAPE (%)", angle=45),
                    ),
                )
                st.plotly_chart(fig_polar, width="stretch")
    elif not run_mape:
        st.info("Click **▶ Run Forecast CV** to compute MAPE across hours.", icon="ℹ️")
