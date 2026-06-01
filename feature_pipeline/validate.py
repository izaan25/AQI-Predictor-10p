"""feature_pipeline/validate.py — Data quality checks before feature store insertion."""
import pandas as pd
from loguru import logger


REQUIRED_COLS    = ["city", "timestamp", "aqi"]
AQI_RANGE        = (0, 500)
PM25_RANGE       = (0, 1000)
HOUR_RANGE       = (0, 23)
HUMIDITY_RANGE   = (0, 100)


def validate_features(df: pd.DataFrame) -> bool:
    """
    Run data quality checks on a feature DataFrame.
    Raises ValueError if critical checks fail.
    Returns True on success, with warnings for non-critical issues.
    """
    errors   = []
    warnings = []

    # ── Completeness ──────────────────────────────────────────
    for col in REQUIRED_COLS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
        elif df[col].isnull().any():
            errors.append(f"Null values in required column: {col}")

    # ── AQI range ─────────────────────────────────────────────
    if "aqi" in df.columns:
        aqi_vals = df["aqi"].dropna()
        out_of_range = aqi_vals[(aqi_vals < AQI_RANGE[0]) | (aqi_vals > AQI_RANGE[1])]
        if not out_of_range.empty:
            errors.append(f"AQI values out of range {AQI_RANGE}: {out_of_range.tolist()}")

    # ── Hour range ────────────────────────────────────────────
    if "hour" in df.columns:
        hour_vals = df["hour"].dropna()
        bad = hour_vals[(hour_vals < HOUR_RANGE[0]) | (hour_vals > HOUR_RANGE[1])]
        if not bad.empty:
            errors.append(f"Hour values out of range {HOUR_RANGE}: {bad.tolist()}")

    # ── PM2.5 range ───────────────────────────────────────────
    if "pm25" in df.columns:
        pm_vals = df["pm25"].dropna()
        bad = pm_vals[(pm_vals < PM25_RANGE[0]) | (pm_vals > PM25_RANGE[1])]
        if not bad.empty:
            warnings.append(f"PM2.5 values outside expected range {PM25_RANGE}: {bad.tolist()}")

    # ── Nulls in optional fields ──────────────────────────────
    optional = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    for col in optional:
        if col in df.columns and df[col].isnull().any():
            warnings.append(f"Null in optional column '{col}' — station may not report this pollutant")

    # ── Drift detection: PM2.5 spike ─────────────────────────
    if "pm25" in df.columns and "aqi_roll_mean_24h" in df.columns:
        for _, row in df.iterrows():
            pm25 = row.get("pm25")
            roll = row.get("aqi_roll_mean_24h")
            if pm25 and roll and roll > 0:
                pct_change = abs(pm25 - roll) / roll * 100
                if pct_change > 50:
                    warnings.append(f"PM2.5 drift: current={pm25:.1f} vs 24h avg={roll:.1f} ({pct_change:.0f}% deviation)")

    # ── Report ────────────────────────────────────────────────
    for w in warnings:
        logger.warning(f"[VALIDATE] ⚠ {w}")

    if errors:
        for e in errors:
            logger.error(f"[VALIDATE] ✗ {e}")
        raise ValueError(f"Data validation failed with {len(errors)} error(s): {errors[0]}")

    logger.info(f"[VALIDATE] ✓ All checks passed ({len(warnings)} warning(s))")
    return True
