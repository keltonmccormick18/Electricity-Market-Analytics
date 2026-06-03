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
import io
import time
import zipfile
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tomllib

import pandas as pd
import requests
import duckdb
from dotenv import load_dotenv

from constants import PRICE_TYPE_DAY_AHEAD


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

# ── ISO price-API configuration ────────────────────────────────────────────────
# Real day-ahead LMP, fetched directly from each ISO (not EIA).
# These replace the old TI-as-price proxy.

# CAISO OASIS — public, no key. Rate-limited to ~1 request / 5s, max 31 days/request.
CAISO_OASIS_URL = "http://oasis.caiso.com/oasisapi/SingleZip"
CAISO_HUB       = "TH_NP15_GEN-APND"   # NP15 trading hub as the CISO reference price
CAISO_DELAY     = 5.0                   # OASIS throttles aggressively

# PJM Data Miner 2 — requires a free subscription key (Ocp-Apim-Subscription-Key).
# Register at https://dataminer2.pjm.com → My Account → API key.
PJM_API_URL  = "https://api.pjm.com/api/v1/da_hrl_lmps"
PJM_HUB_PNODE = 51217                   # Western Hub as the PJM reference price
PJM_PAGE_SIZE = 50000                   # max rowCount per request

# NYISO — public market data, no key. Monthly ZIP archives of daily zonal LBMP.
# Each zip (named by the month's first day) holds one CSV per day.
NYISO_DAMLBMP_URL = "http://mis.nyiso.com/public/csv/damlbmp"   # /{YYYYMM01}damlbmp_zone_csv.zip
NYISO_ZONE        = "N.Y.C."            # NYC zone as the NYIS reference price
NYISO_TZ          = "America/New_York"  # timestamps are Eastern prevailing time → convert to UTC

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

def eia_get(endpoint: str, params: dict, retries: int = 5) -> dict:
    """
    Make a single paginated GET request to the EIA v2 API.
    Retries up to *retries* times with exponential backoff on timeouts or 5xx errors.
    """
    params["api_key"] = API_KEY
    url = f"{BASE_URL}/{endpoint}/data/"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = 2 ** attempt
            log.warning("Request timeout/connection error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, retries, wait, e)
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code >= 500:
                wait = 2 ** attempt
                log.warning("EIA 5xx error (attempt %d/%d), retrying in %ds", attempt + 1, retries, wait)
                time.sleep(wait)
            else:
                raise
    raise requests.exceptions.RetryError(f"EIA API failed after {retries} retries: {url}")


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
    Dispatch real day-ahead LMP ingestion to the per-ISO fetchers.

    Each ISO publishes prices through its own API (EIA does not carry hourly
    wholesale LMP), so there is no shared endpoint. We pull a representative
    trading hub per region and normalize to (hour, region_id, price_type,
    price_usd_mwh) with price_type = 'day_ahead_lmp'.

    TIME ZONES: all sources share one clock (UTC). CAISO returns GMT, PJM is
    fetched as datetime_beginning_utc, EIA demand/generation timestamps are also
    UTC (verified — see _parse_eia_hour), and NYISO's Eastern-prevailing stamps
    are converted to UTC in its fetcher. Joins on `hour` align the same real
    instant across tables, so no offset correction is needed.
    """
    total = 0
    if "CISO" in regions:
        total += ingest_prices_caiso(con, date_from, date_to)
    if "PJM" in regions:
        total += ingest_prices_pjm(con, date_from, date_to)
    if "NYIS" in regions:
        total += ingest_prices_nyiso(con, date_from, date_to)
    return total


def ingest_prices_caiso(con: duckdb.DuckDBPyConnection,
                        date_from: str,
                        date_to: str) -> int:
    """
    Day-ahead LMP for the CISO region from the CAISO OASIS API (PRC_LMP / DAM).

    Public, no key required. Throttled to ~1 req/5s and capped at 31 days per
    request, so we chunk the range monthly. Returns a ZIP containing one CSV;
    the price lives in the 'MW' column where LMP_TYPE == 'LMP'.
    """
    total_inserted = 0
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in _month_chunks(date_from, date_to):
        log.info("Fetching CAISO DAM LMP: %s → %s  (node %s)",
                 chunk_start, chunk_end, CAISO_HUB)
        params = {
            "queryname":     "PRC_LMP",
            "version":       "1",
            "startdatetime": _caiso_dt(chunk_start),
            "enddatetime":   _caiso_dt(chunk_end),
            "market_run_id": "DAM",
            "node":          CAISO_HUB,
            "resultformat":  "6",          # CSV
        }
        try:
            resp = requests.get(CAISO_OASIS_URL, params=params, timeout=120)
            resp.raise_for_status()
        except requests.HTTPError as e:
            log.error("CAISO HTTP error %s → %s: %s", chunk_start, chunk_end, e)
            _log_ingestion(con, "prices_caiso", "CISO", chunk_start, chunk_end, 0, "error", str(e))
            time.sleep(CAISO_DELAY)
            continue

        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile:
            # OASIS returns a non-ZIP error/throttle page instead of data
            log.warning("CAISO returned non-ZIP for %s → %s (no data or throttled)",
                        chunk_start, chunk_end)
            _log_ingestion(con, "prices_caiso", "CISO", chunk_start, chunk_end, 0, "empty")
            time.sleep(CAISO_DELAY)
            continue

        member = zf.namelist()[0]
        if not member.endswith(".csv"):
            # OASIS returns an XML error report instead of a CSV when the request
            # is rejected (e.g. 1000 = no data for selection, 1004 = range too large).
            # Pull the actual code/description so the log says *why* it failed.
            reason = _caiso_error(zf.read(member))
            log.warning("CAISO error for %s → %s: %s", chunk_start, chunk_end, reason)
            _log_ingestion(con, "prices_caiso", "CISO", chunk_start, chunk_end, 0, "error", reason)
            time.sleep(CAISO_DELAY)
            continue

        raw = pd.read_csv(zf.open(member))
        raw = raw[raw["LMP_TYPE"] == "LMP"]
        if raw.empty:
            time.sleep(CAISO_DELAY)
            continue

        out = pd.DataFrame({
            "hour":          pd.to_datetime(raw["INTERVALSTARTTIME_GMT"], utc=True).dt.tz_localize(None),
            "region_id":     "CISO",
            "price_type":    PRICE_TYPE_DAY_AHEAD,
            "price_usd_mwh": raw["MW"].astype(float),
        })
        frames.append(out)
        time.sleep(CAISO_DELAY)

    if frames:
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["hour", "region_id", "price_type"])
        con.execute("INSERT OR REPLACE INTO fact_prices SELECT * FROM df")
        total_inserted = len(df)
        _log_ingestion(con, "prices_caiso", "CISO", date_from, date_to, total_inserted, "success")
        log.info("  inserted %d CAISO price rows", total_inserted)

    return total_inserted


def ingest_prices_pjm(con: duckdb.DuckDBPyConnection,
                      date_from: str,
                      date_to: str) -> int:
    """
    Day-ahead LMP for the PJM region from PJM Data Miner 2 (da_hrl_lmps).

    Requires PJM_API_KEY (Ocp-Apim-Subscription-Key). We pull the Western Hub
    pnode as the PJM reference price and paginate via startRow/rowCount using
    the X-TotalRows response header. total_lmp_da is the day-ahead LMP in $/MWh.
    """
    key = os.getenv("PJM_API_KEY")
    if not key:
        log.warning("PJM_API_KEY not set — skipping PJM prices. "
                    "Register at https://dataminer2.pjm.com for a free key.")
        return 0

    headers = {"Ocp-Apim-Subscription-Key": key}
    # PJM range filter format: 'start to end' in EPT (Eastern Prevailing Time).
    date_filter = f"{date_from} 00:00 to {date_to} 23:59"

    rows: list[dict] = []
    start_row = 1
    while True:
        params = {
            "rowCount":               PJM_PAGE_SIZE,
            "startRow":               start_row,
            "datetime_beginning_ept": date_filter,
            "pnode_id":               PJM_HUB_PNODE,
            "fields":                 "datetime_beginning_utc,pnode_id,total_lmp_da",
        }
        try:
            resp = requests.get(PJM_API_URL, headers=headers, params=params, timeout=120)
            resp.raise_for_status()
        except requests.HTTPError as e:
            log.error("PJM HTTP error (startRow %d): %s", start_row, e)
            _log_ingestion(con, "prices_pjm", "PJM", date_from, date_to, 0, "error", str(e))
            return 0

        payload = resp.json()
        items = payload.get("items", [])
        if not items:
            break
        rows.extend(items)

        total = int(resp.headers.get("X-TotalRows", len(rows)))
        log.info("  fetched %d / %d PJM price rows (startRow %d)", len(rows), total, start_row)
        start_row += PJM_PAGE_SIZE
        if start_row > total:
            break
        time.sleep(REQUEST_DELAY)

    if not rows:
        log.warning("No PJM price data returned")
        return 0

    df = pd.DataFrame(
        [(_parse_pjm_utc(r["datetime_beginning_utc"]), "PJM", PRICE_TYPE_DAY_AHEAD, r.get("total_lmp_da"))
         for r in rows],
        columns=["hour", "region_id", "price_type", "price_usd_mwh"],
    ).drop_duplicates(subset=["hour", "region_id", "price_type"])

    con.execute("INSERT OR REPLACE INTO fact_prices SELECT * FROM df")
    _log_ingestion(con, "prices_pjm", "PJM", date_from, date_to, len(df), "success")
    log.info("  inserted %d PJM price rows", len(df))
    return len(df)


def ingest_prices_nyiso(con: duckdb.DuckDBPyConnection,
                        date_from: str,
                        date_to: str) -> int:
    """
    Day-ahead zonal LBMP for the NYIS region from NYISO public market data.

    Public, no key. Data is published as monthly ZIP archives (named by the
    month's first day), each containing one daily CSV. We take the N.Y.C. zone
    as the NYIS reference price.

    TIME ZONE: NYISO 'Time Stamp' is Eastern *prevailing* time (DST-aware), unlike
    CAISO (GMT) and PJM (UTC). We convert America/New_York → UTC so NYIS shares
    the same clock as every other table (ambiguous fall-back hour inferred from
    order; nonexistent spring-forward hour shifted forward).
    """
    total_inserted = 0
    frames: list[pd.DataFrame] = []

    for ym in _month_firsts(date_from, date_to):
        url = f"{NYISO_DAMLBMP_URL}/{ym}damlbmp_zone_csv.zip"
        log.info("Fetching NYISO DAM LBMP month %s (zone %s)", ym, NYISO_ZONE)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
        except requests.HTTPError as e:
            log.error("NYISO HTTP error for %s: %s", ym, e)
            _log_ingestion(con, "prices_nyiso", "NYIS", ym, ym, 0, "error", str(e))
            time.sleep(REQUEST_DELAY)
            continue

        if resp.content[:2] != b"PK":
            log.warning("NYISO returned non-ZIP for %s (no data?)", ym)
            _log_ingestion(con, "prices_nyiso", "NYIS", ym, ym, 0, "empty")
            time.sleep(REQUEST_DELAY)
            continue

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        day_frames = [
            pd.read_csv(zf.open(n)) for n in zf.namelist() if n.endswith(".csv")
        ]
        if not day_frames:
            time.sleep(REQUEST_DELAY)
            continue

        raw = pd.concat(day_frames, ignore_index=True)
        raw = raw[raw["Name"] == NYISO_ZONE].copy()
        if raw.empty:
            time.sleep(REQUEST_DELAY)
            continue

        # EPT local → UTC (naive), matching the single-clock convention
        ts_local = pd.to_datetime(raw["Time Stamp"], format="%m/%d/%Y %H:%M")
        ts_utc = (
            ts_local.dt.tz_localize(NYISO_TZ, ambiguous="infer", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        out = pd.DataFrame({
            "hour":          ts_utc,
            "region_id":     "NYIS",
            "price_type":    PRICE_TYPE_DAY_AHEAD,
            "price_usd_mwh": raw["LBMP ($/MWHr)"].astype(float),
        })
        frames.append(out)
        time.sleep(REQUEST_DELAY)

    if frames:
        df = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["hour", "region_id", "price_type"]
        )
        con.execute("INSERT OR REPLACE INTO fact_prices SELECT * FROM df")
        total_inserted = len(df)
        _log_ingestion(con, "prices_nyiso", "NYIS", date_from, date_to, total_inserted, "success")
        log.info("  inserted %d NYISO price rows", total_inserted)

    return total_inserted


# ── Utility helpers ───────────────────────────────────────────────────────────

def _caiso_dt(date_str: str) -> str:
    """'YYYY-MM-DD' → CAISO OASIS GMT timestamp 'YYYYMMDDT00:00-0000'."""
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%dT00:00-0000")


def _caiso_error(xml_bytes: bytes) -> str:
    """
    Extract 'ERR_CODE: ERR_DESC' from a CAISO OASIS error report.
    Falls back to a short raw snippet if the expected tags aren't present.
    """
    import re
    text = xml_bytes.decode("utf-8", "replace")
    code = re.search(r"<m:ERR_CODE>(.*?)</m:ERR_CODE>", text)
    desc = re.search(r"<m:ERR_DESC>(.*?)</m:ERR_DESC>", text)
    if code or desc:
        return f"{code.group(1) if code else '?'}: {desc.group(1) if desc else '?'}"
    return text[:200].strip()


def _month_firsts(date_from: str, date_to: str):
    """
    Yield 'YYYYMM01' strings for every calendar month spanning
    [date_from, date_to]. Used to address NYISO monthly ZIP archives.
    """
    cur = datetime.strptime(date_from, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(date_to, "%Y-%m-%d")
    while cur <= end:
        yield cur.strftime("%Y%m01")
        cur = (cur.replace(year=cur.year + 1, month=1) if cur.month == 12
               else cur.replace(month=cur.month + 1))


def _parse_pjm_utc(period: str) -> str:
    """
    PJM datetime_beginning_utc → naive 'YYYY-MM-DD HH:MM:SS'.
    PJM returns e.g. '2024-01-15T14:00:00' (already UTC); we store naive UTC.
    """
    return pd.to_datetime(period).strftime("%Y-%m-%d %H:%M:%S")


def _month_chunks(date_from: str, date_to: str):
    """
    Yield (start, end) 'YYYY-MM-DD' pairs ≤ 30 days apart, covering
    [date_from, date_to]. CAISO OASIS rejects PRC_LMP ranges of 31+ days
    (error 1004) once the inclusive GMT boundary is applied, so we use 30.
    """
    start = datetime.strptime(date_from, "%Y-%m-%d")
    end   = datetime.strptime(date_to, "%Y-%m-%d")
    while start < end:
        chunk_end = min(start + timedelta(days=30), end)
        yield start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        start = chunk_end


def _parse_eia_hour(period: str) -> str:
    """
    Convert EIA period string to a naive 'YYYY-MM-DD HH:00:00' timestamp.

    EIA's hourly rto endpoints return periods as 'YYYY-MM-DDTHH' in **UTC**
    (no tz suffix). This was verified empirically: CISO solar generation peaks
    at stored hour 20, i.e. 20:00 UTC ≈ local noon — so these timestamps are
    UTC, not local BA time. CAISO/PJM prices are also stored in UTC, so all
    `hour` columns share one clock and join directly with no offset.
    (Diagnostic: solar-peak anchor + demand–price cross-correlation peaks at
    lag +2h, the intrinsic economic phase, not a 7–8h clock offset.)
    """
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