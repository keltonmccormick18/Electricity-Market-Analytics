FUEL_COLORS = {
    "SUN": "#FFD700", "WND": "#4FC3F7", "WAT": "#1565C0",
    "NUC": "#E53935", "NG":  "#FB8C00", "COL": "#546E7A",
    "BAT": "#43A047", "OIL": "#6D4C41", "OTH": "#90A4AE",
}

FUEL_LABELS = {
    "SUN": "Solar",    "WND": "Wind",         "WAT": "Hydro",
    "NUC": "Nuclear",  "NG":  "Natural Gas",   "COL": "Coal",
    "BAT": "Battery",  "OIL": "Oil",           "OTH": "Other",
}

# Bottom → top stacking order (baseload at bottom, variable at top)
FUEL_ORDER = ["NUC", "WAT", "COL", "OIL", "OTH", "NG", "BAT", "WND", "SUN"]

REGION_COLORS = {
    "CISO": "#1f77b4", "PJM":  "#ff7f0e", "ERCO": "#2ca02c",
    "MISO": "#d62728", "NYIS": "#9467bd", "ISNE": "#8c564b",
}

REGIONS      = ["CISO", "PJM", "ERCO", "MISO", "NYIS", "ISNE"]
PRICE_REGIONS = ["CISO", "PJM", "NYIS", "ISNE", "ERCO"]   # EIA price coverage (TI proxy)

DOW_LABELS = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

# fact_prices.price_type value for day-ahead LMP. Written by ingestion, filtered by consumers.
PRICE_TYPE_DAY_AHEAD = "day_ahead_lmp"
