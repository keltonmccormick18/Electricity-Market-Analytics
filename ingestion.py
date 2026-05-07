"""
EIA Open Data API v2 — Ingestion Pipeline
==========================================
Fetches hourly electricity demand, generation by fuel type,
and wholesale prices for major US balancing authorities.

Usage:
    # First run: full historical load (2014–present)
    python ingest.py --mode full --regions CISO PJM ERCO

    # Daily incremental update (last 7 days)
    python ingest.py --mode incremental

    # Single region test
    python ingest.py --mode full --regions CISO --start 2024-01-01 --end 2024-03-31

Setup:
    pip install duckdb requests python-dotenv
    Create a .env file: EIA_API_KEY=your_key_here
    Get a free key at: https://www.eia.gov/opendata/register.php
"""

import os
import time
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tomllib

import pandas as pd
import requests
import duckdb
from dotenv import load_dotenv


def _load_streamlit_secrets(path: str = ".streamlit/secrets.toml") -> None:
    """Inject .streamlit/secrets.toml values into os.environ (if not already set)."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("rb") as f:
        secrets = tomllib.load(f)
    for key, value in secrets.items():
        if isinstance(value, str) and key not in os.environ:
            os.environ[key] = value


load_dotenv()
_load_streamlit_secrets()

# ── Configuration ──────────────────────────────────────────────────────────────

API_KEY   = os.getenv("EIA_API_KEY", "DEMO_KEY")   # DEMO_KEY is rate-limited to ~30 req/day
BASE_URL  = "https://api.eia.gov/v2"
DB_PATH   = Path("data/electricity.duckdb")
SCHEMA    = Path("schema.sql")

# Balancing authorities to ingest by default
DEFAULT_REGIONS = ["CISO", "PJM", "MISO", "ERCO", "NYIS", "ISNE", "SWPP"]

# EIA pagination limit (max rows per request)
PAGE_SIZE = 5000

# Polite delay between API calls (seconds) — avoid rate limiting
REQUEST_DELAY = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ── Database setup ─────────────────────────────────────────────────────────────

def get_db(target: str = str(DB_PATH)) -> duckdb.DuckDBPyConnection:
    """
    Open (or create) the database and apply schema.

    *target* is either:
      - a local file path  (default: data/electricity.duckdb)
      - a MotherDuck URI   (e.g. md:energy)

    For MotherDuck, MOTHERDUCK_TOKEN must be set in the environment.
    """
    is_motherduck = target.startswith("md:")
    if is_motherduck:
        token = os.getenv("MOTHERDUCK_TOKEN")
        if not token:
            raise SystemExit(
                "MOTHERDUCK_TOKEN env var is required for MotherDuck connections.\n"
                "  export MOTHERDUCK_TOKEN=eyJhbGc..."
            )
        con = duckdb.connect(target, config={"motherduck_token": token})
        log.info("Connected to MotherDuck: %s", target)
    else:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(path))
        log.info("Connected to local DB: %s", path.resolve())

    if SCHEMA.exists():
        con.execute(SCHEMA.read_text())
        log.info("Schema applied from %s", SCHEMA)
    return con


# ── EIA API helpers ────────────────────────────────────────────────────────────

def eia_get(endpoint: str, params: dict) -> dict:
    """
    Make a single paginated GET request to the EIA v2 API.
    Raises on HTTP errors. Returns the parsed JSON dict.
    """
    params["api_key"] = API_KEY
    url = f"{BASE_URL}/{endpoint}/data/"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def eia_fetch_all(endpoint: str, params: dict) -> list[dict]:
    """
    Fetch ALL pages for a given endpoint + params combo.
    EIA v2 uses offset-based pagination via params['offset'].
    Returns a flat list of row dicts.
    """
    all_rows = []
    offset   = 0
    params   = {**params, "length": PAGE_SIZE}

    while True:
        params["offset"] = offset
        data = eia_get(endpoint, params)

        response_data = data.get("response", {})
        rows          = response_data.get("data", [])
        total         = int(response_data.get("total", 0))

        if not rows:
            break

        all_rows.extend(rows)
        log.info("  fetched %d / %d rows (offset %d)", len(all_rows), total, offset)

        offset += PAGE_SIZE
        if offset >= total:
            break

        time.sleep(REQUEST_DELAY)

    return all_rows


# ── Demand ingestion ──────────────────────────────────────────────────────────

def ingest_demand(con: duckdb.DuckDBPyConnection,
                  regions: list[str],
                  date_from: str,
                  date_to: str) -> int:
    """
    Fetch hourly actual demand and day-ahead forecast for each region.

    EIA endpoint: electricity/rto/region-data
    Series IDs:
        D  = actual demand (MWh)
        DF = day-ahead demand forecast (MWh)
    """
    total_inserted = 0

    for region in regions:
        log.info("Fetching demand: %s  %s → %s", region, date_from, date_to)

        params = {
            "frequency":        "hourly",
            "data[0]":          "value",
            "facets[respondent][]": region,
            "facets[type][]":   ["D", "DF"],
            "start":            date_from,
            "end":              date_to,
            "sort[0][column]":  "period",
            "sort[0][direction]": "asc",
        }

        try:
            rows = eia_fetch_all("electricity/rto/region-data", params)
        except requests.HTTPError as e:
            log.error("HTTP error for %s demand: %s", region, e)
            _log_ingestion(con, "demand", region, date_from, date_to, 0, "error", str(e))
            continue

        if not rows:
            log.warning("No demand data returned for %s", region)
            continue

        # Pivot D and DF into separate columns
        demand_map: dict[str, dict] = {}
        for row in rows:
            hour = row["period"]           # e.g. "2024-01-15T14"
            if hour not in demand_map:
                demand_map[hour] = {"demand_mwh": None, "demand_forecast": None}
            if row["type"] == "D":
                demand_map[hour]["demand_mwh"] = row.get("value")
            elif row["type"] == "DF":
                demand_map[hour]["demand_forecast"] = row.get("value")

        df = pd.DataFrame(
            [(_parse_eia_hour(h), region, v["demand_mwh"], v["demand_forecast"])
             for h, v in demand_map.items()],
            columns=["hour", "region_id", "demand_mwh", "demand_forecast"],
        )
        con.execute("INSERT OR REPLACE INTO fact_demand SELECT * FROM df")
        total_inserted += len(df)
        _log_ingestion(con, "demand", region, date_from, date_to, len(df), "success")
        log.info("  inserted %d demand rows for %s", len(df), region)

    return total_inserted


# ── Generation ingestion ──────────────────────────────────────────────────────

def ingest_generation(con: duckdb.DuckDBPyConnection,
                      regions: list[str],
                      date_from: str,
                      date_to: str) -> int:
    """
    Fetch hourly net generation by fuel type for each region.

    EIA endpoint: electricity/rto/fuel-type-data
    """
    total_inserted = 0

    # EIA fuel type codes
    fuel_types = ["SUN", "WND", "WAT", "NUC", "NG", "COL", "OIL", "OTH", "BAT"]

    for region in regions:
        log.info("Fetching generation: %s  %s → %s", region, date_from, date_to)

        params = {
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": region,
            "start":                date_from,
            "end":                  date_to,
            "sort[0][column]":      "period",
            "sort[0][direction]":   "asc",
        }

        try:
            rows = eia_fetch_all("electricity/rto/fuel-type-data", params)
        except requests.HTTPError as e:
            log.error("HTTP error for %s generation: %s", region, e)
            _log_ingestion(con, "generation", region, date_from, date_to, 0, "error", str(e))
            continue

        if not rows:
            log.warning("No generation data returned for %s", region)
            continue

        df = pd.DataFrame(
            [(_parse_eia_hour(row["period"]), region, row.get("fueltype", "OTH"), row.get("value"))
             for row in rows if row.get("fueltype") in fuel_types],
            columns=["hour", "region_id", "fuel_id", "generation_mwh"],
        )
        con.execute("INSERT OR REPLACE INTO fact_generation SELECT * FROM df")
        total_inserted += len(df)
        _log_ingestion(con, "generation", region, date_from, date_to, len(df), "success")
        log.info("  inserted %d generation rows for %s", len(df), region)

    return total_inserted


# ── Price ingestion ───────────────────────────────────────────────────────────

def ingest_prices(con: duckdb.DuckDBPyConnection,
                  regions: list[str],
                  date_from: str,
                  date_to: str) -> int:
    """
    Fetch hourly day-ahead LMP prices.
    Note: EIA has limited price coverage — NYIS, ISNE, PJM have the best data.

    EIA endpoint: electricity/rto/region-data  (type=TI = total interchange, 
    prices available via electricity/wholesale-prices for some nodes)
    """
    # EIA's wholesale price endpoint covers fewer regions than demand/generation.
    # We use the day-ahead price series where available.
    price_regions = [r for r in regions if r in ("NYIS", "ISNE", "PJM", "CISO")]

    if not price_regions:
        log.info("No price-eligible regions in selection, skipping prices")
        return 0

    total_inserted = 0

    for region in price_regions:
        log.info("Fetching prices: %s  %s → %s", region, date_from, date_to)

        params = {
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": region,
            "facets[type][]":       ["TI"],   # total interchange as price proxy
            "start":                date_from,
            "end":                  date_to,
            "sort[0][column]":      "period",
            "sort[0][direction]":   "asc",
        }

        try:
            rows = eia_fetch_all("electricity/rto/region-data", params)
        except requests.HTTPError as e:
            log.error("HTTP error for %s prices: %s", region, e)
            continue

        if rows:
            df = pd.DataFrame(
                [(_parse_eia_hour(row["period"]), region, "day_ahead", row.get("value"))
                 for row in rows],
                columns=["hour", "region_id", "price_type", "price_usd_mwh"],
            )
            con.execute("INSERT OR REPLACE INTO fact_prices SELECT * FROM df")
            total_inserted += len(df)
            _log_ingestion(con, "prices", region, date_from, date_to, len(df), "success")
            log.info("  inserted %d price rows for %s", len(df), region)

    return total_inserted


# ── Utility helpers ───────────────────────────────────────────────────────────

def _parse_eia_hour(period: str) -> str:
    """
    Convert EIA period string to ISO 8601 UTC timestamp.
    EIA returns periods as 'YYYY-MM-DDTHH' in local BA time (no tz info).
    We store as-is and handle tz in analysis queries.
    """
    # EIA format: '2024-01-15T14' → append ':00:00+00:00' for DuckDB TIMESTAMPTZ
    if len(period) == 13:           # 'YYYY-MM-DDTHH'
        return period + ":00:00"
    return period


def _log_ingestion(con, endpoint, region, date_from, date_to,
                   rows_inserted, status, error_msg=None):
    con.execute(
        """
        INSERT INTO ingestion_log
            (endpoint, region_id, date_from, date_to, rows_inserted, status, error_msg)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [endpoint, region, date_from, date_to, rows_inserted, status, error_msg]
    )


def _get_incremental_start(con: duckdb.DuckDBPyConnection) -> str:
    """Return the day after the latest demand record, or 7 days ago if empty."""
    result = con.execute(
        "SELECT MAX(hour)::DATE FROM fact_demand"
    ).fetchone()[0]

    if result:
        return str(result)
    return str((datetime.now(timezone.utc) - timedelta(days=7)).date())


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EIA electricity data ingestion pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="incremental",
        help="full = fetch from --start to --end; incremental = fetch since last record"
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=DEFAULT_REGIONS,
        help="Space-separated EIA balancing authority codes e.g. CISO PJM ERCO"
    )
    parser.add_argument(
        "--start",
        default="2014-01-01",
        help="Start date for full mode (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        default=str(datetime.now(timezone.utc).date()),
        help="End date for full mode (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--db",
        default=str(DB_PATH),
        help=(
            "Database target — local file path (default: data/electricity.duckdb) "
            "or MotherDuck URI (e.g. md:energy). "
            "MotherDuck requires MOTHERDUCK_TOKEN in the environment."
        ),
    )
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip price ingestion (faster for initial load)"
    )
    args = parser.parse_args()

    con = get_db(args.db)

    if args.mode == "incremental":
        date_from = _get_incremental_start(con)
        date_to   = str(datetime.now(timezone.utc).date())
        log.info("Incremental mode: %s → %s", date_from, date_to)
    else:
        date_from = args.start
        date_to   = args.end
        log.info("Full mode: %s → %s across %d regions", date_from, date_to, len(args.regions))

    log.info("Regions: %s", ", ".join(args.regions))
    log.info("DB: %s", args.db)

    if API_KEY == "DEMO_KEY":
        log.warning("Using DEMO_KEY — rate limited to ~30 requests/day. "
                    "Register at https://www.eia.gov/opendata/register.php for a free key.")

    # Run ingestion
    d = ingest_demand(con, args.regions, date_from, date_to)
    g = ingest_generation(con, args.regions, date_from, date_to)
    p = 0 if args.skip_prices else ingest_prices(con, args.regions, date_from, date_to)

    log.info("Done. demand=%d  generation=%d  prices=%d rows inserted", d, g, p)

    # Quick sanity check
    counts = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM fact_demand)     AS demand_rows,
            (SELECT COUNT(*) FROM fact_generation) AS generation_rows,
            (SELECT COUNT(*) FROM fact_prices)     AS price_rows,
            (SELECT COUNT(*) FROM ingestion_log)   AS log_entries
    """).fetchone()
    log.info("DB totals — demand: %d  generation: %d  prices: %d  log: %d", *counts)

    con.close()


if __name__ == "__main__":
    main()