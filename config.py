"""config.py — Central configuration. Works 100% free in local mode."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
DB_PATH    = DATA_DIR / "features.db"      # SQLite — used in local mode
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "latest").mkdir(exist_ok=True)

# ── API credentials ──────────────────────────────────────────────────────────
AQICN_TOKEN    = os.getenv("AQICN_TOKEN", "")
OW_API_KEY     = os.getenv("OW_API_KEY", "")
HOPSWORKS_KEY  = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJ = os.getenv("HOPSWORKS_PROJECT", "pearls-aqi")

# ── Storage mode: "local" (SQLite) or "hopsworks" ───────────────────────────
STORAGE_MODE = os.getenv("STORAGE_MODE", "local").lower()
# Auto-upgrade disabled — always use STORAGE_MODE from environment

# ── Target city ──────────────────────────────────────────────────────────────
TARGET_CITY = os.getenv("TARGET_CITY", "karachi")

CITY_COORDS = {
    "karachi":  {"lat": 24.8607, "lon": 67.0011, "aqicn_id": "@7012"},
    "lahore":   {"lat": 31.5497, "lon": 74.3436, "aqicn_id": "@7011"},
    "delhi":    {"lat": 28.6139, "lon": 77.2090, "aqicn_id": "@7015"},
    "beijing":  {"lat": 39.9042, "lon": 116.407, "aqicn_id": "@1451"},
    "london":   {"lat": 51.5074, "lon": -0.1278, "aqicn_id": "@5724"},
}

# ── API endpoints ────────────────────────────────────────────────────────────
AQICN_BASE = "https://api.waqi.info"
OW_BASE    = "https://api.openweathermap.org/data/2.5"
OW_UNITS   = "metric"

# ── Hopsworks feature group settings ─────────────────────────────────────────
FG_NAME        = "fg_aqi_features"
FG_VERSION     = 3
FG_PRIMARY_KEY = ["city", "timestamp"]
FG_EVENT_TIME  = "timestamp"

# ── Model training ───────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
LAG_HOURS    = [1, 3, 6, 12, 24]

# ── Retry ────────────────────────────────────────────────────────────────────
MAX_RETRIES  = 3
BACKOFF_BASE = 1.0

# ── AQI scale ────────────────────────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",                    "#00e400"),
    (51,  100, "Moderate",                "#ffff00"),
    (101, 150, "Unhealthy for Sensitive", "#ff7e00"),
    (151, 200, "Unhealthy",               "#ff0000"),
    (201, 300, "Very Unhealthy",          "#8f3f97"),
    (301, 500, "Hazardous",               "#7e0023"),
]

def aqi_category(aqi) -> tuple[str, str]:
    """Return (label, hex_color) for an AQI value."""
    try:
        aqi = int(aqi)
    except (TypeError, ValueError):
        return "Unknown", "#888888"
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "#7e0023"
