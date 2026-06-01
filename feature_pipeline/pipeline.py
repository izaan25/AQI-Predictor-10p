"""feature_pipeline/pipeline.py — Hourly pipeline orchestrator."""
import pandas as pd
from loguru import logger
from config import TARGET_CITY
from feature_pipeline.fetch import fetch_all
from feature_pipeline.engineer import compute_features
from feature_pipeline.store import push_features, pull_latest_features


def run_feature_pipeline(city: str = TARGET_CITY) -> dict:
    """
    Run the complete hourly feature pipeline:
    1. Fetch real-time data from AQICN + OpenWeather
    2. Load recent history for lag features
    3. Compute engineered features
    4. Save to store (local SQLite or Hopsworks)
    """
    logger.info(f"=== Feature Pipeline START  city={city} ===")

    # 1. Fetch
    raw = fetch_all(city)

    # 2. History
    try:
        history = pull_latest_features(city, n_hours=48)
    except Exception as e:
        logger.warning(f"Could not load history (first run?): {e}")
        history = pd.DataFrame(columns=["aqi", "pm25", "pm10"])

    # 3. Compute
    features = compute_features(raw, history)

    # 4. Store
    push_features(pd.DataFrame([features]))

    logger.info(f"=== Feature Pipeline DONE  AQI={features['aqi']} ===")
    return features


if __name__ == "__main__":
    result = run_feature_pipeline()
    print(f"\n✓ Done. AQI={result['aqi']}  city={result['city']}")
