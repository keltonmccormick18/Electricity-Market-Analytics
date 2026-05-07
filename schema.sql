-- Electricity market schema
-- Applied automatically by get_db() on every connection (IF NOT EXISTS = safe to re-run)

CREATE TABLE IF NOT EXISTS fact_demand (
    hour            TIMESTAMP    NOT NULL,
    region_id       VARCHAR      NOT NULL,
    demand_mwh      DOUBLE,
    demand_forecast DOUBLE,
    PRIMARY KEY (hour, region_id)
);

CREATE TABLE IF NOT EXISTS fact_generation (
    hour            TIMESTAMP    NOT NULL,
    region_id       VARCHAR      NOT NULL,
    fuel_id         VARCHAR      NOT NULL,
    generation_mwh  DOUBLE,
    PRIMARY KEY (hour, region_id, fuel_id)
);

CREATE TABLE IF NOT EXISTS fact_prices (
    hour            TIMESTAMP    NOT NULL,
    region_id       VARCHAR      NOT NULL,
    price_type      VARCHAR      NOT NULL,
    price_usd_mwh   DOUBLE,
    PRIMARY KEY (hour, region_id, price_type)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    endpoint        VARCHAR,
    region_id       VARCHAR,
    date_from       VARCHAR,
    date_to         VARCHAR,
    rows_inserted   INTEGER,
    status          VARCHAR,
    error_msg       VARCHAR,
    created_at      TIMESTAMP    DEFAULT current_timestamp
);
