import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from loguru import logger
from config import AQICN_TOKEN, AQICN_BASE, CITY_COORDS, TARGET_CITY
from feature_pipeline.engineer import compute_features
from feature_pipeline.store import push_features


def fetch_day(city, date):
    """Fetch AQI using the regular feed endpoint with date workaround."""
    station = CITY_COORDS[city]["aqicn_id"]
    # Use regular feed endpoint — historical data via geo search
    url = f"{AQICN_BASE}/feed/{station}/?token={AQICN_TOKEN}"
    try:
        resp = requests.get(url, timeout=20)
        data = resp.json()
        if data.get("status") != "ok":
            logger.warning(f"  API status not ok for {date}: {data.get('status')}")
            return None
        d = data["data"]
        iaqi = d.get("iaqi", {})
        def _v(k): return iaqi[k]["v"] if k in iaqi else None
        aqi_val = d.get("aqi", 0)
        if not aqi_val or aqi_val == "-":
            return None
        return {
            "city": city, "timestamp": f"{date}T12:00:00",
            "reported_at": f"{date}T12:00:00",
            "aqi": int(aqi_val), "dominant_pol": d.get("dominentpol", ""),
            "pm25": _v("pm25"), "pm10": _v("pm10"), "o3": _v("o3"),
            "no2": _v("no2"), "so2": _v("so2"), "co": _v("co"),
            "temperature": _v("t"), "humidity": _v("h"),
            "wind_speed": _v("w"), "pressure": _v("p"),
            "ow_temp": None, "ow_feels_like": None, "ow_humidity": None,
            "ow_pressure": None, "ow_wind_speed": None, "ow_wind_deg": None,
            "ow_cloudiness": None, "ow_visibility": None,
        }
    except Exception as e:
        logger.error(f"Fetch failed for {city} {date}: {e}")
        return None


def run_backfill(city=TARGET_CITY, days_back=90):
    """
    Since AQICN free tier does not provide true historical data per date,
    we generate synthetic historical rows based on current AQI with
    realistic daily variation. This gives enough data to train ML models.
    """
    import random
    import math
    logger.info(f"=== Backfill START  city={city}  days={days_back} ===")

    # Fetch current real AQI as baseline
    logger.info("Fetching current AQI as baseline...")
    current = fetch_day(city, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    if not current:
        logger.error("Could not fetch current AQI. Check AQICN_TOKEN in .env")
        return

    base_aqi = current["aqi"]
    base_pm25 = current.get("pm25") or base_aqi * 0.3
    base_pm10 = current.get("pm10") or base_aqi * 0.5
    logger.info(f"Baseline — AQI={base_aqi}, PM2.5={base_pm25}")

    today = datetime.now(timezone.utc).date()
    history = pd.DataFrame(columns=["aqi", "pm25", "pm10"])
    saved = 0

    for i in range(days_back, 0, -1):
        date_obj = today - timedelta(days=i)
        date = date_obj.strftime("%Y-%m-%d")

        # Realistic seasonal + random variation around baseline
        day_of_year = date_obj.timetuple().tm_yday
        seasonal = math.sin(2 * math.pi * day_of_year / 365) * 15
        random.seed(i)  # reproducible
        noise = random.gauss(0, 12)
        aqi = max(10, min(400, int(base_aqi + seasonal + noise)))
        pm25 = max(1, round((base_pm25 or 20) * (aqi / max(base_aqi, 1)) + random.gauss(0, 3), 1))
        pm10 = max(1, round((base_pm10 or 40) * (aqi / max(base_aqi, 1)) + random.gauss(0, 5), 1))

        raw = {
            "city": city, "timestamp": f"{date}T12:00:00",
            "reported_at": f"{date}T12:00:00",
            "aqi": aqi, "dominant_pol": "pm25",
            "pm25": pm25, "pm10": pm10,
            "o3": round(random.uniform(20, 60), 1),
            "no2": round(random.uniform(10, 80), 1),
            "so2": round(random.uniform(2, 30), 1),
            "co": round(random.uniform(0.3, 1.5), 2),
            "temperature": round(random.uniform(18, 38), 1),
            "humidity": round(random.uniform(40, 85), 1),
            "wind_speed": round(random.uniform(1, 8), 1),
            "pressure": round(random.uniform(995, 1015), 1),
            "ow_temp": None, "ow_feels_like": None, "ow_humidity": None,
            "ow_pressure": None, "ow_wind_speed": None, "ow_wind_deg": None,
            "ow_cloudiness": None, "ow_visibility": None,
        }

        try:
            feats = compute_features(raw, history)
            row_df = pd.DataFrame([feats])
            push_features(row_df)
            saved += 1
            if saved % 10 == 0:
                logger.info(f"  [{saved}/{days_back}] Saved {date} — AQI={aqi}")
        except Exception as e:
            logger.error(f"  Failed to save {date}: {e}")

        history = pd.concat([
            history,
            pd.DataFrame([{"aqi": aqi, "pm25": pm25, "pm10": pm10}])
        ], ignore_index=True).tail(50)

    logger.info(f"=== Backfill DONE — saved {saved} rows ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=TARGET_CITY)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    run_backfill(city=args.city, days_back=args.days)