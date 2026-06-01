import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger
from config import STORAGE_MODE, DB_PATH, HOPSWORKS_KEY, HOPSWORKS_PROJ, FG_NAME, FG_VERSION


def _local_push(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    with sqlite3.connect(str(DB_PATH)) as conn:
        df.to_sql("features", conn, if_exists="append", index=False)
        conn.commit()
    logger.info(f"[LOCAL] Saved {len(df)} row(s) to {DB_PATH}")


def _local_pull(city=None, n_rows=None):
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(DB_PATH)) as conn:
        try:
            if city:
                df = pd.read_sql(
                    "SELECT * FROM features WHERE city=? ORDER BY timestamp",
                    conn, params=(city,)
                )
            else:
                df = pd.read_sql("SELECT * FROM features ORDER BY timestamp", conn)
        except Exception:
            return pd.DataFrame()
    if n_rows:
        df = df.tail(n_rows)
    return df.reset_index(drop=True)


def push_features(df):
    _local_push(df)


def pull_training_data(city=None):
    logger.info(f"Pulling training data [local]...")
    return _local_pull(city=city)


def pull_latest_features(city, n_hours=48):
    logger.info(f"Pulling last {n_hours}h for {city} [local]...")
    return _local_pull(city=city, n_rows=n_hours)