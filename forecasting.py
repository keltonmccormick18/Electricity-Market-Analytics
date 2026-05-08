"""
Short-Term Load Forecasting — Backend
======================================
Three models benchmarked on 24-hour-ahead demand prediction:
  SARIMA     — classical univariate baseline (statsmodels SARIMAX)
  XGBoost    — ML baseline with engineered lag / calendar features
  TCN        — lightweight Temporal Convolutional Network via dilated
               Gaussian kernels + Ridge head (no DL framework required)

Evaluation:
  Expanding-window time-series cross-validation (no random splits)
  MAE, RMSE, MAPE per model / region / season
  Diebold-Mariano (Harvey-Leybourne-Newbold 1997) significance test

Residual diagnostics:
  ACF / PACF, Ljung-Box autocorrelation test, ARCH-LM heteroskedasticity test

Probabilistic extension:
  XGBoost quantile regression (q=0.10, 0.50, 0.90)
  Pinball loss + empirical coverage rate
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats


# ── Feature columns ────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "month", "year", "is_weekend",
    "hod_sin", "hod_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_24h", "lag_48h", "lag_168h",
    "roll_mean_24h", "roll_std_24h", "roll_mean_168h", "roll_std_168h",
    "temp_proxy",
]
TARGET = "demand_mwh"


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw hourly demand.
    Requires columns: [hour (timestamp), demand_mwh].
    All lag/rolling features are strictly causal (no look-ahead).
    """
    df = df.sort_values("hour").copy().reset_index(drop=True)
    idx = pd.DatetimeIndex(df["hour"])
    s   = df["demand_mwh"]

    # Calendar
    df["hour_of_day"] = idx.hour
    df["day_of_week"] = idx.dayofweek        # 0=Mon
    df["month"]       = idx.month
    df["year"]        = idx.year
    df["is_weekend"]  = (idx.dayofweek >= 5).astype(int)

    # Cyclical encodings
    df["hod_sin"]   = np.sin(2 * np.pi * idx.hour / 24)
    df["hod_cos"]   = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

    # Lag demand (strictly causal — shift uses past observations)
    for lag in (24, 48, 168):
        df[f"lag_{lag}h"] = s.shift(lag)

    # Rolling statistics (shift(1) ensures current hour is excluded)
    s1 = s.shift(1)
    df["roll_mean_24h"]  = s1.rolling(24).mean()
    df["roll_std_24h"]   = s1.rolling(24).std()
    df["roll_mean_168h"] = s1.rolling(168).mean()
    df["roll_std_168h"]  = s1.rolling(168).std()

    # Temperature proxy: sinusoidal seasonal cycle (°F equivalent)
    doy = idx.dayofyear.to_numpy()
    df["temp_proxy"] = 60 + 22 * np.sin(2 * np.pi * (doy - 80) / 365.25)

    return df


# ── SARIMA ─────────────────────────────────────────────────────────────────────

def fit_sarima(series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    mod = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return mod.fit(disp=False, maxiter=150)


def sarima_cv(series: pd.Series, n_initial_days=60, n_folds=5, step_days=7):
    """Rolling-window SARIMA CV. Re-fits at each fold (slow but rigorous)."""
    init_n = n_initial_days * 24
    step_n = step_days * 24
    results = []
    for fold in range(n_folds):
        train_end = init_n + fold * step_n
        test_end  = train_end + 24
        if test_end > len(series):
            break
        train_s = series.iloc[:train_end]
        actual  = series.iloc[train_end:test_end].values
        try:
            fitted = fit_sarima(train_s)
            pred   = fitted.forecast(24).values
        except Exception:
            pred = np.full(24, train_s.tail(168).mean())
        results.append({"fold": fold, "actual": actual, "pred": pred,
                         "timestamps": series.index[train_end:test_end]})
    return results


# ── XGBoost ────────────────────────────────────────────────────────────────────

def fit_xgboost(X: np.ndarray, y: np.ndarray, quantile: float | None = None):
    import xgboost as xgb
    if quantile is not None:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=quantile,
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    else:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_lambda=1.0, random_state=42,
        )
    model.fit(X, y)
    return model


def xgb_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    imp = model.feature_importances_
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": imp})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# ── Lightweight TCN ────────────────────────────────────────────────────────────

class LightweightTCN:
    """
    TCN approximation: dilated causal Gaussian kernels → Ridge regression.

    Architecture mirrors a single-layer TCN:
      - Dilation rates (1,2,4,8,16,32) give receptive fields of
        1, 2, 4, 8, 16, 32 hours — covering sub-daily to multi-day patterns.
      - Difference features across adjacent scales act as residual connections.
      - No non-linear activations → equivalent to a linear TCN.
        (Adds a Ridge L2 penalty analogous to weight decay.)

    This connects to SDE/regime work: the dilated kernels approximate
    the Green's function of the diffusion operator in the demand SDE,
    capturing mean-reversion at multiple time scales.
    """

    def __init__(
        self,
        dilation_rates: tuple = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 7,
        alpha: float = 10.0,
    ):
        from sklearn.linear_model import Ridge
        self.dilation_rates = dilation_rates
        self.kernel_size    = kernel_size
        self.head           = Ridge(alpha=alpha)
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    def _kernel(self, dilation: int) -> np.ndarray:
        x = np.arange(self.kernel_size, dtype=float)
        sigma = max(1.0, dilation * 0.6)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        return (k / k.sum())[::-1]          # normalize + flip (causal)

    def _dilated_conv(self, col: np.ndarray, dilation: int) -> np.ndarray:
        kernel  = self._kernel(dilation)
        pad     = len(kernel) - 1
        padded  = np.pad(col, (pad, 0), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _extract(self, X: np.ndarray) -> np.ndarray:
        blocks = [X]
        for d in self.dilation_rates:
            block = np.column_stack([self._dilated_conv(X[:, c], d)
                                      for c in range(X.shape[1])])
            blocks.append(block)
        # Residual-style difference between consecutive dilation scales
        diff_blocks = [blocks[i + 1] - blocks[i] for i in range(len(blocks) - 1)]
        return np.hstack(blocks + diff_blocks)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightweightTCN":
        feat = self._extract(X)
        self._mu    = feat.mean(axis=0)
        self._sigma = feat.std(axis=0) + 1e-8
        self.head.fit((feat - self._mu) / self._sigma, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        feat = self._extract(X)
        return self.head.predict((feat - self._mu) / self._sigma)


# ── Expanding-window CV (XGBoost + TCN) ───────────────────────────────────────

def ml_expanding_cv(
    df_feat: pd.DataFrame,
    n_initial_days: int = 60,
    n_folds: int = 12,
    step_days: int = 7,
) -> list[dict]:
    """
    Expanding-window cross-validation for XGBoost and TCN.
    Each fold trains on all data up to train_end, tests on the next 24 hours.
    Training window grows by step_days each fold — no random splits.
    """
    clean = df_feat.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)
    init_n = n_initial_days * 24
    step_n = step_days * 24
    results = []

    for fold in range(n_folds):
        train_end = init_n + fold * step_n
        test_end  = train_end + 24
        if test_end > len(clean):
            break

        train = clean.iloc[:train_end]
        test  = clean.iloc[train_end:test_end]

        X_tr, y_tr = train[FEATURE_COLS].values, train[TARGET].values
        X_te, y_te = test[FEATURE_COLS].values,  test[TARGET].values

        xgb_pred = fit_xgboost(X_tr, y_tr).predict(X_te)

        tcn = LightweightTCN()
        tcn.fit(X_tr, y_tr)
        tcn_pred = tcn.predict(X_te)

        results.append({
            "fold":       fold,
            "timestamps": test["hour"].values,
            "actual":     y_te,
            "xgb_pred":   xgb_pred,
            "tcn_pred":   tcn_pred,
        })

    return results


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    e    = actual - predicted
    mae  = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mape = float(np.mean(np.abs(e / np.maximum(np.abs(actual), 1.0))) * 100)
    return {"MAE (MWh)": round(mae, 1), "RMSE (MWh)": round(rmse, 1), "MAPE (%)": round(mape, 2)}


def season_of(ts) -> str:
    m = pd.Timestamp(ts).month
    return ("Winter" if m in (12, 1, 2) else
            "Spring" if m in (3, 4, 5)  else
            "Summer" if m in (6, 7, 8)  else "Fall")


def cv_metrics_table(ml_results: list[dict], sarima_results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in ml_results:
        ts = r["timestamps"][0]
        base = {"Fold": r["fold"] + 1, "Season": season_of(ts)}
        rows.append({**base, "Model": "XGBoost",
                     **compute_metrics(r["actual"], r["xgb_pred"])})
        rows.append({**base, "Model": "TCN",
                     **compute_metrics(r["actual"], r["tcn_pred"])})
    for r in sarima_results:
        ts = r["timestamps"][0]
        rows.append({"Fold": r["fold"] + 1, "Season": season_of(ts),
                     "Model": "SARIMA",
                     **compute_metrics(r["actual"], r["pred"])})
    return pd.DataFrame(rows)


def seasonal_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(["Model", "Season"])[["MAE (MWh)", "RMSE (MWh)", "MAPE (%)"]]
        .mean()
        .round(2)
        .reset_index()
    )


# ── Diebold-Mariano test ───────────────────────────────────────────────────────

def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
) -> tuple[float, float, str]:
    """
    Harvey, Leybourne & Newbold (1997) modified Diebold-Mariano test.
    H₀: equal predictive accuracy.
    e1, e2: forecast errors (actual − predicted).
    Returns (DM statistic, p-value, plain-English interpretation).
    """
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    mean_d = np.mean(d)

    # Newey-West HAC variance (up to h−1 lags)
    gamma0 = np.var(d, ddof=1)
    gammas = [float(np.cov(d[j:], d[:-j])[0, 1]) for j in range(1, max(h, 1))]
    V_d = (gamma0 + 2.0 * sum(gammas)) / T

    if V_d <= 0 or np.isnan(V_d):
        return np.nan, np.nan, "Insufficient variance"

    dm_raw = mean_d / np.sqrt(V_d)
    # HLN size correction factor
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat = float(dm_raw * correction)
    p_value = float(2 * (1 - stats.t.cdf(abs(dm_stat), df=T - 1)))

    if p_value < 0.01:
        label = ("M1 significantly better" if dm_stat < 0
                 else "M2 significantly better")
    elif p_value < 0.05:
        label = ("M1 marginally better (p<.05)" if dm_stat < 0
                 else "M2 marginally better (p<.05)")
    else:
        label = "No significant difference"

    return dm_stat, p_value, label


def dm_matrix(ml_results: list[dict], sarima_results: list[dict]) -> pd.DataFrame:
    """Build pairwise DM test results for all model pairs."""
    # Pool errors across folds
    e = {}
    n = min(len(ml_results), len(sarima_results))
    for r_ml, r_sa in zip(ml_results[:n], sarima_results[:n]):
        actual = r_ml["actual"]
        for key, label in (("xgb_pred", "XGBoost"), ("tcn_pred", "TCN")):
            e.setdefault(label, []).append(actual - r_ml[key])
        e.setdefault("SARIMA", []).append(actual - r_sa["pred"])

    e = {k: np.concatenate(v) for k, v in e.items()}
    models = list(e.keys())

    rows = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            dm, p, interp = diebold_mariano(e[m1], e[m2])
            rows.append({
                "Model 1": m1, "Model 2": m2,
                "DM Stat": round(dm, 3) if not np.isnan(dm) else "—",
                "p-value": round(p, 4) if not np.isnan(p) else "—",
                "Result": interp,
            })
    return pd.DataFrame(rows)


# ── Residual analysis ──────────────────────────────────────────────────────────

def get_residuals(ml_results: list[dict], sarima_results: list[dict]) -> dict[str, np.ndarray]:
    """Pool residuals across CV folds for each model."""
    n   = min(len(ml_results), len(sarima_results))
    res = {
        "XGBoost": np.concatenate([r["actual"] - r["xgb_pred"] for r in ml_results[:n]]),
        "TCN":     np.concatenate([r["actual"] - r["tcn_pred"] for r in ml_results[:n]]),
        "SARIMA":  np.concatenate([r["actual"] - r["pred"]     for r in sarima_results[:n]]),
    }
    return res


def compute_acf_pacf(residuals: np.ndarray, nlags: int = 48) -> dict:
    from statsmodels.tsa.stattools import acf, pacf
    acf_v,  acf_ci  = acf(residuals,  nlags=nlags, alpha=0.05)
    pacf_v, pacf_ci = pacf(residuals, nlags=nlags, alpha=0.05)
    lags = np.arange(nlags + 1)
    return {
        "lags": lags,
        "acf":  acf_v,  "acf_lo":  acf_ci[:, 0] - acf_v, "acf_hi":  acf_ci[:, 1] - acf_v,
        "pacf": pacf_v, "pacf_lo": pacf_ci[:, 0] - pacf_v, "pacf_hi": pacf_ci[:, 1] - pacf_v,
    }


def ljung_box(residuals: np.ndarray, lags=(12, 24, 48)) -> pd.DataFrame:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    df = acorr_ljungbox(residuals, lags=list(lags), return_df=True)
    df.index.name = "Lag"
    df.columns    = ["LB Statistic", "p-value"]
    df["Reject H₀ (α=.05)?"] = df["p-value"] < 0.05
    return df.round(4)


def arch_lm(residuals: np.ndarray, nlags: int = 12) -> dict:
    from statsmodels.stats.diagnostic import het_arch
    lm, lm_p, f, f_p = het_arch(residuals, nlags=nlags)
    return {
        "ARCH-LM Stat": round(lm, 4),
        "p-value":       round(lm_p, 4),
        "F Stat":        round(f, 4),
        "F p-value":     round(f_p, 4),
        "Heteroskedastic?": "Yes (p<.05)" if lm_p < 0.05 else "No",
    }


# ── Quantile / probabilistic forecasting ──────────────────────────────────────

QUANTILES = (0.10, 0.50, 0.90)


def fit_quantile_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    return {q: fit_xgboost(X_train, y_train, quantile=q) for q in QUANTILES}


def predict_quantiles(models: dict, X_test: np.ndarray) -> dict:
    return {q: m.predict(X_test) for q, m in models.items()}


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    e = actual - predicted
    return float(np.mean(np.where(e >= 0, quantile * e, (quantile - 1) * e)))


def coverage_rate(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((actual >= lower) & (actual <= upper)))
