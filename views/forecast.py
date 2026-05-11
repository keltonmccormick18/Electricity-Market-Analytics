"""
Demand Forecast — Streamlit UI
================================
Tab 1 · Model Comparison   : train SARIMA / XGBoost / TCN, compare metrics
Tab 2 · Residual Analysis  : ACF/PACF, Ljung-Box, ARCH-LM
Tab 3 · Probabilistic      : XGBoost quantile regression (10 / 50 / 90 %)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import db
from constants import REGIONS, REGION_COLORS
import forecasting as fc

# ── Data loader ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_demand(region: str, train_years: int) -> pd.DataFrame | None:
    con = db.get_connection()
    result = db.run_query(con, f"""
        SELECT hour, demand_mwh
        FROM fact_demand
        WHERE region_id = '{region}'
          AND hour >= (SELECT MAX(hour) FROM fact_demand)
                      - INTERVAL '{train_years} years'
          AND demand_mwh > 0
        ORDER BY hour
    """)
    if result.error or result.df is None or result.df.empty:
        return None
    df = result.df.copy()
    df["hour"] = pd.to_datetime(df["hour"])
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings", anchor=False)
    region       = st.selectbox("Region", REGIONS, index=0)
    train_years  = st.slider("Training window (years)", 1, 5, 2)
    n_folds_ml   = st.slider("CV folds (XGB / TCN)", 4, 16, 8)
    n_folds_sa   = st.slider("CV folds (SARIMA)", 2, 8, 3,
                              help="SARIMA refits each fold — keep this low")

# ── Page header ───────────────────────────────────────────────────────────────

st.header("Demand Forecast", anchor=False, divider="gray")

tab_cmp, tab_res, tab_prob = st.tabs(
    ["📊 Model Comparison", "🔬 Residual Analysis", "📈 Probabilistic Forecast"]
)

# ── Session-state keys ────────────────────────────────────────────────────────

_SK = "fc_results"   # dict keyed by (region, train_years)

def _cache_key():
    return (region, train_years, n_folds_ml, n_folds_sa)


def _get_cached():
    return st.session_state.get(_SK, {}).get(_cache_key())


def _set_cached(val):
    st.session_state.setdefault(_SK, {})[_cache_key()] = val


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

with tab_cmp:
    cached = _get_cached()

    run_btn = st.button("▶ Train All Models", type="primary",
                         disabled=(cached is not None))
    if cached is not None:
        st.caption("Results cached — adjust settings above to retrain.")
        if st.button("🔄 Clear cache & retrain"):
            st.session_state.get(_SK, {}).pop(_cache_key(), None)
            st.rerun()

    if run_btn:
        df_raw = load_demand(region, train_years)
        if df_raw is None:
            st.error("No demand data found for this region / window.")
        else:
            with st.status("Training models…", expanded=True) as status:
                st.write("Engineering features…")
                df_feat = fc.engineer_features(df_raw)

                st.write(f"Running SARIMA CV ({n_folds_sa} folds)…")
                sarima_series = df_raw.set_index("hour")["demand_mwh"]
                sarima_results = fc.sarima_cv(
                    sarima_series, n_folds=n_folds_sa, n_initial_days=60
                )

                st.write(f"Running XGBoost + TCN expanding CV ({n_folds_ml} folds)…")
                ml_results = fc.ml_expanding_cv(
                    df_feat, n_folds=n_folds_ml, n_initial_days=60
                )

                st.write("Computing metrics and DM tests…")
                metrics_df   = fc.cv_metrics_table(ml_results, sarima_results)
                seasonal_df  = fc.seasonal_summary(metrics_df)
                dm_df        = fc.dm_matrix(ml_results, sarima_results)
                residuals    = fc.get_residuals(ml_results, sarima_results)

                # Quantile models using last fold's train split
                clean = df_feat.dropna(subset=fc.FEATURE_COLS + [fc.TARGET])
                split = int(len(clean) * 0.80)
                X_tr = clean.iloc[:split][fc.FEATURE_COLS].values
                y_tr = clean.iloc[:split][fc.TARGET].values
                X_te = clean.iloc[split:][fc.FEATURE_COLS].values
                y_te = clean.iloc[split:][fc.TARGET].values
                q_models = fc.fit_quantile_models(X_tr, y_tr)
                q_preds  = fc.predict_quantiles(q_models, X_te)
                q_times  = clean.iloc[split:]["hour"].values

                _set_cached({
                    "ml_results":    ml_results,
                    "sarima_results": sarima_results,
                    "metrics_df":    metrics_df,
                    "seasonal_df":   seasonal_df,
                    "dm_df":         dm_df,
                    "residuals":     residuals,
                    "q_preds":       q_preds,
                    "y_te":          y_te,
                    "q_times":       q_times,
                })
                status.update(label="Training complete!", state="complete")

    cached = _get_cached()
    if cached:
        ml_results     = cached["ml_results"]
        sarima_results = cached["sarima_results"]
        metrics_df     = cached["metrics_df"]
        seasonal_df    = cached["seasonal_df"]
        dm_df          = cached["dm_df"]

        # ── 24h forecast overlay — last fold shared across all models ────────
        # SARIMA has fewer folds than ML; use the last index present in both
        # so all three traces cover the same 24-hour test window.
        shared_idx = min(len(ml_results), len(sarima_results)) - 1
        last     = ml_results[shared_idx]
        sa_match = sarima_results[shared_idx]
        ts       = pd.to_datetime(last["timestamps"])

        color = REGION_COLORS.get(region, "#4fc3f7")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=last["actual"],
                                  name="Actual", line=dict(color=color)))
        fig.add_trace(go.Scatter(x=ts, y=last["xgb_pred"],
                                  name="XGBoost", line=dict(color="#ff7f0e", dash="dash")))
        fig.add_trace(go.Scatter(x=ts, y=last["tcn_pred"],
                                  name="TCN", line=dict(color="#2ca02c", dash="dot")))
        sa_ts = pd.to_datetime(sa_match["timestamps"])
        fig.add_trace(go.Scatter(x=sa_ts, y=sa_match["pred"],
                                  name="SARIMA", line=dict(color="#9467bd", dash="longdash")))
        fig.update_layout(
            template="plotly_dark", height=320,
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="Demand (MWh)",
        )
        st.subheader("24-hour ahead forecast — last shared CV fold", anchor=False)
        st.plotly_chart(fig, width="stretch")

        # ── Overall metrics summary ────────────────────────────────────────────
        st.subheader("Cross-validation metrics (mean across folds)", anchor=False)
        overall = (
            metrics_df.groupby("Model")[["MAE (MWh)", "RMSE (MWh)", "MAPE (%)"]]
            .mean().round(2).reset_index()
            .sort_values("MAE (MWh)")
        )
        st.dataframe(overall, hide_index=True, width="stretch")

        # ── Seasonal breakdown ────────────────────────────────────────────────
        with st.expander("Seasonal breakdown"):
            season_order = ["Winter", "Spring", "Summer", "Fall"]
            sd = seasonal_df.copy()
            sd["Season"] = pd.Categorical(sd["Season"], categories=season_order, ordered=True)
            st.dataframe(
                sd.sort_values(["Season", "Model"]).reset_index(drop=True),
                hide_index=True, width="stretch",
            )

        # ── DM test matrix ────────────────────────────────────────────────────
        with st.expander("Diebold-Mariano significance test"):
            st.caption(
                "H₀: equal predictive accuracy.  "
                "DM < 0 ⟹ Model 1 better;  DM > 0 ⟹ Model 2 better."
            )
            st.dataframe(dm_df, hide_index=True, width="stretch")

    elif not run_btn:
        st.info(
            "Click **▶ Train All Models** to run expanding-window CV "
            "and benchmark SARIMA, XGBoost, and TCN.",
            icon="ℹ️",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_res:
    cached = _get_cached()
    if cached is None:
        st.info("Train models in **Model Comparison** first.", icon="ℹ️")
    else:
        residuals = cached["residuals"]
        model_sel = st.selectbox("Model", list(residuals.keys()), key="res_model")
        resid     = residuals[model_sel]

        st.subheader(f"Residuals — {model_sel}", anchor=False)

        # ── ACF / PACF ────────────────────────────────────────────────────────
        ap = fc.compute_acf_pacf(resid, nlags=48)
        lags = ap["lags"]

        col_acf, col_pacf = st.columns(2)
        for col, key, title in (
            (col_acf,  "acf",  "ACF"),
            (col_pacf, "pacf", "PACF"),
        ):
            vals = ap[key]
            lo   = ap[f"{key}_lo"]
            hi   = ap[f"{key}_hi"]
            ci   = 1.96 / np.sqrt(len(resid))

            fig = go.Figure()
            # Confidence band
            fig.add_hrect(y0=-ci, y1=ci,
                           fillcolor="rgba(100,100,100,0.15)", line_width=0)
            # Bars
            colors = ["#e74c3c" if abs(v) > ci else "#4fc3f7" for v in vals]
            for i, (x, y, c) in enumerate(zip(lags, vals, colors)):
                fig.add_trace(go.Bar(x=[x], y=[y], marker_color=c,
                                     showlegend=False, name=""))
            fig.update_layout(
                template="plotly_dark", height=260, title=title,
                margin=dict(t=30, b=20), bargap=0.1,
                yaxis=dict(range=[-0.4, 0.4]),
                xaxis_title="Lag (hours)",
            )
            col.plotly_chart(fig, width="stretch")

        # ── Rolling residual variance ─────────────────────────────────────────
        with st.expander("Rolling residual variance (24h window)"):
            roll_var = pd.Series(resid ** 2).rolling(24).mean().dropna()
            fig_rv = px.line(
                x=np.arange(len(roll_var)), y=roll_var.values,
                labels={"x": "Sample", "y": "Rolling Var (MWh²)"},
                template="plotly_dark",
            )
            fig_rv.update_layout(height=220, margin=dict(t=10, b=10))
            st.plotly_chart(fig_rv, width="stretch")

        # ── Statistical tests ─────────────────────────────────────────────────
        st.subheader("Diagnostic tests", anchor=False)
        col_lb, col_arch = st.columns(2)

        with col_lb:
            st.markdown("**Ljung-Box test** (H₀: no autocorrelation)")
            lb_df = fc.ljung_box(resid, lags=(12, 24, 48))
            lb_df["Reject H₀ (α=.05)?"] = lb_df["Reject H₀ (α=.05)?"].map(
                {True: "✗ Yes", False: "✓ No"}
            )
            st.dataframe(lb_df, width="stretch")

        with col_arch:
            st.markdown("**ARCH-LM test** (H₀: no heteroskedasticity)")
            arch = fc.arch_lm(resid, nlags=12)
            arch_df = pd.DataFrame([arch]).T.reset_index()
            arch_df.columns = ["Statistic", "Value"]
            st.dataframe(arch_df, hide_index=True, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROBABILISTIC FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

with tab_prob:
    cached = _get_cached()
    if cached is None:
        st.info("Train models in **Model Comparison** first.", icon="ℹ️")
    else:
        q_preds = cached["q_preds"]
        y_te    = cached["y_te"]
        q_times = pd.to_datetime(cached["q_times"])

        # Show last N hours for readability
        n_display = st.slider("Hours to display", 48, min(720, len(y_te)), 168, step=24)
        sl = slice(-n_display, None)

        times  = q_times[sl]
        actual = y_te[sl]
        p10    = q_preds[0.10][sl]
        p50    = q_preds[0.50][sl]
        p90    = q_preds[0.90][sl]

        color  = REGION_COLORS.get(region, "#4fc3f7")

        fig = go.Figure()
        # 80% PI shading
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="80% interval",
        ))
        fig.add_trace(go.Scatter(
            x=times, y=actual, name="Actual",
            line=dict(color=color),
        ))
        fig.add_trace(go.Scatter(
            x=times, y=p50, name="Median (q=0.50)",
            line=dict(color="#ff7f0e", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=times, y=p10, name="q=0.10",
            line=dict(color="#aaa", width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=times, y=p90, name="q=0.90",
            line=dict(color="#aaa", width=1, dash="dot"),
        ))
        fig.update_layout(
            template="plotly_dark", height=360,
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="Demand (MWh)",
        )
        st.subheader("Quantile forecast — XGBoost (10th / 50th / 90th percentile)",
                      anchor=False)
        st.plotly_chart(fig, width="stretch")

        # ── Evaluation metrics ────────────────────────────────────────────────
        st.subheader("Probabilistic evaluation", anchor=False)

        full_actual = y_te
        full_p10    = q_preds[0.10]
        full_p50    = q_preds[0.50]
        full_p90    = q_preds[0.90]

        pb_rows = [
            {
                "Quantile": f"q={q:.2f}",
                "Pinball Loss (MWh)": round(fc.pinball_loss(full_actual, q_preds[q], q), 2),
            }
            for q in fc.QUANTILES
        ]
        cov = fc.coverage_rate(full_actual, full_p10, full_p90)

        col_pb, col_cov = st.columns([2, 1])
        col_pb.dataframe(pd.DataFrame(pb_rows), hide_index=True, width="stretch")
        col_cov.metric("80% PI coverage", f"{cov * 100:.1f}%",
                        delta=f"{(cov - 0.80) * 100:+.1f} pp vs target")

        st.caption(
            "Pinball loss is the standard scoring rule for quantile forecasts.  "
            "Coverage rate is the fraction of actuals falling within the 10–90% interval "
            "(ideal ≈ 80%)."
        )
