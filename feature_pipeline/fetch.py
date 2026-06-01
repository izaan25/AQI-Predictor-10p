"""feature_pipeline/fetch.py — Real-time data from AQICN + OpenWeather (both free)."""
import time
import requests
from datetime import datetime, timezone
from loguru import logger
from config import (
    AQICN_TOKEN, AQICN_BASE,
    OW_API_KEY, OW_BASE, OW_UNITS,
    CITY_COORDS, MAX_RETRIES, BACKOFF_BASE,
)


def fetch_aqicn(city: str) -> dict:
    """Fetch live AQI + pollutants from AQICN API (free token)."""
    station_id = CITY_COORDS[city]["aqicn_id"]
    url = f"{AQICN_BASE}/feed/{station_id}/?token={AQICN_TOKEN}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") != "ok":
                raise ValueError(f"AQICN error: {payload.get('data', payload)}")
            return _parse_aqicn(payload["data"], city)
        except Exception as exc:
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"AQICN attempt {attempt+1} failed: {exc}. Retry in {wait}s")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    raise RuntimeError(f"AQICN failed after {MAX_RETRIES} attempts for {city}")


def _parse_aqicn(data: dict, city: str) -> dict:
    iaqi = data.get("iaqi", {})
    def _v(k):
        return iaqi[k]["v"] if k in iaqi else None

    raw_time = data.get("time", {}).get("s", "")
    try:
        reported_at = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        reported_at = datetime.now(timezone.utc)

    return {
        "city":         city,
        "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "reported_at":  reported_at.strftime("%Y-%m-%dT%H:%M:%S"),
        "aqi":          int(data.get("aqi", 0)),
        "dominant_pol": data.get("dominentpol", ""),
        "pm25":         _v("pm25"),
        "pm10":         _v("pm10"),
        "o3":           _v("o3"),
        "no2":          _v("no2"),
        "so2":          _v("so2"),
        "co":           _v("co"),
        "temperature":  _v("t"),
        "humidity":     _v("h"),
        "wind_speed":   _v("w"),
        "pressure":     _v("p"),
    }


def fetch_openweather(city: str) -> dict:
    """Fetch weather features from OpenWeather (free tier: 1M calls/month)."""
    coords = CITY_COORDS[city]
    url = (
        f"{OW_BASE}/weather"
        f"?lat={coords['lat']}&lon={coords['lon']}"
        f"&appid={OW_API_KEY}&units={OW_UNITS}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return _parse_openweather(resp.json())
        except Exception as exc:
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"OpenWeather attempt {attempt+1} failed: {exc}. Retry in {wait}s")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    logger.warning("OpenWeather failed — weather fields will be null")
    return {}


def _parse_openweather(data: dict) -> dict:
    main   = data.get("main", {})
    wind   = data.get("wind", {})
    clouds = data.get("clouds", {})
    return {
        "ow_temp":       main.get("temp"),
        "ow_feels_like": main.get("feels_like"),
        "ow_humidity":   main.get("humidity"),
        "ow_pressure":   main.get("pressure"),
        "ow_wind_speed": wind.get("speed"),
        "ow_wind_deg":   wind.get("deg"),
        "ow_cloudiness": clouds.get("all"),
        "ow_visibility": data.get("visibility"),
    }


def fetch_all(city: str) -> dict:
    """Merge AQICN + OpenWeather into one flat dict."""
    logger.info(f"Fetching AQICN for {city}...")
    aqicn = fetch_aqicn(city)
    logger.info(f"  AQI={aqicn['aqi']}  PM2.5={aqicn['pm25']}")

    logger.info(f"Fetching OpenWeather for {city}...")
    ow = fetch_openweather(city)

    merged = {**ow,**aqicn}
    # Prefer AQICN station weather if available, else OpenWeather
    merged["temperature"] = ow.get("ow_temp")      or aqicn.get("temperature")
    merged["humidity"]    = ow.get("ow_humidity")   or aqicn.get("humidity")
    merged["wind_speed"]  = ow.get("ow_wind_speed") or aqicn.get("wind_speed")
    merged["pressure"]    = ow.get("ow_pressure")   or aqicn.get("pressure")
    return merged


if __name__ == "__main__":
    import json
    from config import TARGET_CITY
    print(json.dumps(fetch_all(TARGET_CITY), indent=2))
