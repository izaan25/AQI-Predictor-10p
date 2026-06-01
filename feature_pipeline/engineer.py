import numpy as np
import pandas as pd
from datetime import datetime, timezone
from loguru import logger


def time_features(ts: datetime) -> dict:
    h, dow, doy, m = ts.hour, ts.weekday(), ts.timetuple().tm_yday, ts.month
    return {
        "hour":        h,
        "day_of_week": dow,
        "day_of_month":ts.day,
        "month":       m,
        "is_weekend":  int(dow >= 5),
        "hour_sin":    np.sin(2 * np.pi * h   / 24),
        "hour_cos":    np.cos(2 * np.pi * h   / 24),
        "dow_sin":     np.sin(2 * np.pi * dow / 7),
        "dow_cos":     np.cos(2 * np.pi * dow / 7),
        "month_sin":   np.sin(2 * np.pi * m   / 12),
        "month_cos":   np.cos(2 * np.pi * m   / 12),
    }


def lag_features(history: pd.DataFrame, lags: list = [1, 3, 6, 12, 24]) -> dict:
    feats, n = {}, len(history)
    has_aqi  = "aqi"  in history.columns
    has_pm25 = "pm25" in history.columns
    for lag in lags:
        feats[f"aqi_lag_{lag}h"]  = history["aqi"].iloc[-lag]  if (has_aqi  and n >= lag) else None
        feats[f"pm25_lag_{lag}h"] = history["pm25"].iloc[-lag] if (has_pm25 and n >= lag) else None
    return feats


def rolling_features(history: pd.DataFrame, windows: list = [6, 12, 24]) -> dict:
    feats   = {}
    has_aqi = "aqi" in history.columns
    n       = len(history)
    for w in windows:
        if has_aqi and n >= 2:
            tail = history["aqi"].tail(w)
            feats[f"aqi_roll_mean_{w}h"] = round(tail.mean(), 2)
            feats[f"aqi_roll_std_{w}h"]  = round(tail.std(), 2)
            feats[f"aqi_roll_max_{w}h"]  = tail.max()
        else:
            feats[f"aqi_roll_mean_{w}h"] = history["aqi"].iloc[-1] if (has_aqi and n > 0) else None
            feats[f"aqi_roll_std_{w}h"]  = 0.0
            feats[f"aqi_roll_max_{w}h"]  = history["aqi"].iloc[-1] if (has_aqi and n > 0) else None
    return feats


def derived_features(raw: dict, history: pd.DataFrame) -> dict:
    feats = {}
    n = len(history)
    if n >= 1 and "aqi" in history.columns:
        prev = history["aqi"].iloc[-1]
        feats["aqi_change_rate"] = raw["aqi"] - prev
        feats["aqi_change_pct"]  = round(((raw["aqi"] - prev) / (prev + 1e-6)) * 100, 2)
    else:
        feats["aqi_change_rate"] = 0.0
        feats["aqi_change_pct"]  = 0.0

    pm25 = raw.get("pm25") or 1.0
    pm10 = raw.get("pm10") or 1.0
    hum  = raw.get("humidity") or raw.get("ow_humidity") or 50.0
    wind = raw.get("wind_speed") or raw.get("ow_wind_speed") or 0.0

    feats["pm25_pm10_ratio"] = round(pm25 / (pm10 + 1e-6), 4)
    feats["pollution_load"]  = sum(raw.get(k) or 0 for k in ["pm25","pm10","o3","no2","so2"])
    feats["humidity_pm25"]   = round(hum * pm25 / 100.0, 2)
    feats["wind_aqi_ratio"]  = round(raw["aqi"] / (wind + 1.0), 2)
    return feats


def compute_features(raw: dict, history: pd.DataFrame) -> dict:
    ts_str = raw.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

    feats = {"city": raw["city"], "timestamp": ts_str}

    for col in [
        "aqi","pm25","pm10","o3","no2","so2","co",
        "temperature","humidity","wind_speed","pressure",
        "ow_temp","ow_feels_like","ow_humidity","ow_pressure",
        "ow_wind_speed","ow_wind_deg","ow_cloudiness","ow_visibility",
    ]:
        feats[col] = raw.get(col)

    feats.update(time_features(ts))
    feats.update(lag_features(history))
    feats.update(rolling_features(history))
    feats.update(derived_features(raw, history))

    feats["aqi_next_24h"] = None
    feats["aqi_next_48h"] = None
    feats["aqi_next_72h"] = None

    logger.debug(f"Computed {len(feats)} features for {raw['city']} @ {ts_str}")
    return feats