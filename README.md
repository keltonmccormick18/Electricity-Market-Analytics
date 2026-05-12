# Electricity Market Analytics

An interactive dashboard for exploring wholesale electricity markets across six US grid operators — built on live EIA data, DuckDB, and Streamlit.

**Live app → [electricity-market-analytics.streamlit.app](https://electricity-market-analytics.streamlit.app)**  
**Full write-up → [Report]([url](https://www.overleaf.com/read/bthwwwmvqcvc#8bcc56)) (Overleaf)**

---

## Pages

| Page | What it shows |
|---|---|
| **Home** | Live data timestamp, region coverage, record counts |
| **Generation Mix** | Stacked area of fuel contributions over time by region |
| **Price Analytics** | Hour × day-of-week heatmap, spike table, merit order scatter |
| **EDA** | STL/MSTL decomposition, duck curve, peak profiling, spike characterisation, forecast MAPE by hour |
| **SQL Explorer** | Live DuckDB query editor with schema browser and pre-built queries |
| **Demand Forecast** | SARIMA / XGBoost / TCN benchmark, residual diagnostics, quantile regression |

---

## Run Locally

**Prerequisites:** Python 3.11+, an [EIA API key](https://www.eia.gov/opendata/register.php)

```bash
git clone https://github.com/keltonmccormick18/Electricity-Market-Analytics.git
cd Electricity-Market-Analytics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
EIA_API_KEY = "your_key_here"
# Optional — omit to use a local DuckDB file instead
# MOTHERDUCK_TOKEN = "your_token_here"
```

Ingest data (writes to `energy.duckdb` locally):

```bash
python ingestion.py --regions CISO PJM ERCO --start 2020-01-01
```

Run the dashboard:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data store | DuckDB (local) / MotherDuck (cloud) |
| Ingestion | Python + EIA API v2, pandas bulk inserts |
| Dashboard | Streamlit ≥ 1.36, Plotly |
| Forecasting | statsmodels (SARIMA), XGBoost, scikit-learn (Ridge/TCN) |
| Deployment | Streamlit Community Cloud |

---

## Repository Structure

```
├── app.py               # st.navigation router
├── db.py                # DuckDB connection layer (local + MotherDuck)
├── ingestion.py         # EIA API ingestion with retry + bulk inserts
├── schema.sql           # CREATE TABLE IF NOT EXISTS for all four tables
├── constants.py         # Fuel colours, region lists, label maps
├── forecasting.py       # SARIMA / XGBoost / TCN CV + evaluation backend
├── eda.py               # EDA computation backend
├── report.tex           # Full write-up (Overleaf) — motivation, findings, methodology
└── views/
    ├── home.py
    ├── generation_mix.py
    ├── price_analytics.py
    ├── eda.py
    ├── sql_explorer.py
    └── forecast.py
```
