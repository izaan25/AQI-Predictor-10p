import json
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from config import MODELS_DIR, TEST_SIZE, RANDOM_STATE, TARGET_CITY

MODEL_DIR = MODELS_DIR / "latest"

FEATURE_COLS = [
    "hour","day_of_week","day_of_month","month","is_weekend",
    "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
    "aqi","pm25","pm10","o3","no2","so2","co",
    "temperature","humidity","wind_speed","pressure",
    "aqi_lag_1h","aqi_lag_3h","aqi_lag_6h","aqi_lag_12h","aqi_lag_24h",
    "pm25_lag_1h","pm25_lag_6h","pm25_lag_24h",
    "aqi_roll_mean_6h","aqi_roll_mean_12h","aqi_roll_mean_24h",
    "aqi_roll_std_6h","aqi_roll_max_24h",
    "aqi_change_rate","aqi_change_pct","pm25_pm10_ratio",
    "pollution_load","humidity_pm25","wind_aqi_ratio",
]


def _clean(X):
    return pd.DataFrame(X).fillna(0).astype(float)


def _prep(df, target):
    df = df.dropna(subset=[target])
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = _clean(df[avail])
    y = df[target].astype(float)
    return X, y, avail


def _metrics(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    logger.info(f"  [{name}]  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"rmse": round(rmse,3), "mae": round(mae,3), "r2": round(r2,4)}


def train_rf(Xtr, ytr, Xte, yte):
    Xtr, Xte = _clean(Xtr), _clean(Xte)
    m = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
    )
    m.fit(Xtr, ytr)
    return m, _metrics(yte, m.predict(Xte), "RandomForest")


def train_ridge(Xtr, ytr, Xte, yte):
    Xtr, Xte = _clean(Xtr), _clean(Xte)
    sc = StandardScaler()
    m  = Ridge(alpha=10.0)
    m.fit(sc.fit_transform(Xtr), ytr)
    return {"model": m, "scaler": sc}, _metrics(yte, m.predict(sc.transform(Xte)), "Ridge")


def train_xgb(Xtr, ytr, Xte, yte):
    Xtr, Xte = _clean(Xtr), _clean(Xte)
    m = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, tree_method="hist", verbosity=0
    )
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    return m, _metrics(yte, m.predict(Xte), "XGBoost")


def train_lstm(Xtr, ytr, Xte, yte):
    import tensorflow as tf
    from tensorflow import keras
    Xtr, Xte = _clean(Xtr), _clean(Xte)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).reshape(-1, 1, Xtr.shape[1])
    Xte_s = sc.transform(Xte).reshape(-1, 1, Xte.shape[1])
    m = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(1, Xtr.shape[1]), return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    m.fit(
        Xtr_s, ytr,
        validation_data=(Xte_s, yte),
        epochs=50, batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    preds = m.predict(Xte_s, verbose=0).flatten()
    return {"model": m, "scaler": sc}, _metrics(yte, preds, "LSTM")


def save_all(models, metrics, feat_cols):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, obj in models.items():
        if name == "lstm":
            obj["model"].save(str(MODEL_DIR / "lstm_model.keras"))
            joblib.dump(obj["scaler"], MODEL_DIR / "lstm_scaler.pkl")
        elif name == "ridge":
            joblib.dump(obj["model"],  MODEL_DIR / "ridge_model.pkl")
            joblib.dump(obj["scaler"], MODEL_DIR / "ridge_scaler.pkl")
        else:
            joblib.dump(obj, MODEL_DIR / f"{name}_model.pkl")
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump({
            "feature_cols": feat_cols,
            "metrics": metrics,
            "trained_at": pd.Timestamp.now().isoformat()
        }, f, indent=2)
    logger.info(f"Models saved to {MODEL_DIR}")


def load_models():
    models, meta = {}, {}
    rf_p = MODEL_DIR / "rf_model.pkl"
    if rf_p.exists():
        models["rf"] = joblib.load(rf_p)
    rp = MODEL_DIR / "ridge_model.pkl"
    if rp.exists():
        models["ridge"] = {
            "model":  joblib.load(rp),
            "scaler": joblib.load(MODEL_DIR / "ridge_scaler.pkl")
        }
    xp = MODEL_DIR / "xgb_model.pkl"
    if xp.exists():
        models["xgb"] = joblib.load(xp)
    lp = MODEL_DIR / "lstm_model.keras"
    if lp.exists():
        try:
            import tensorflow as tf
            models["lstm"] = {
            "model":  tf.keras.models.load_model(str(lp)),
            "scaler": joblib.load(MODEL_DIR / "lstm_scaler.pkl")
        }
        except Exception:
            pass  # TensorFlow not installed — LSTM skipped silently
    mp = MODEL_DIR / "metadata.json"
    if mp.exists():
        with open(mp) as f:
            meta = json.load(f)
    return models, meta


def run_training(city=TARGET_CITY):
    from feature_pipeline.store import pull_training_data
    logger.info(f"=== Training Pipeline START  city={city} ===")

    df = pull_training_data(city=city)
    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} rows — need at least 30. "
            "Run:  py -3.11 -m feature_pipeline.backfill  first."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi_next_24h"] = df["aqi"].shift(-24)
    df["aqi_next_48h"] = df["aqi"].shift(-48)
    df["aqi_next_72h"] = df["aqi"].shift(-72)

    X, y, feat_cols = _prep(df, "aqi_next_24h")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    logger.info(f"Train={len(Xtr)}  Test={len(Xte)}  Features={len(feat_cols)}")

    models, metrics = {}, {}

    logger.info("Training Random Forest...")
    models["rf"], metrics["rf"] = train_rf(Xtr, ytr, Xte, yte)

    logger.info("Training Ridge Regression...")
    models["ridge"], metrics["ridge"] = train_ridge(Xtr, ytr, Xte, yte)

    logger.info("Training XGBoost...")
    models["xgb"], metrics["xgb"] = train_xgb(Xtr, ytr, Xte, yte)

    logger.info("Training TensorFlow LSTM...")
    try:
        models["lstm"], metrics["lstm"] = train_lstm(Xtr, ytr, Xte, yte)
    except Exception as e:
        logger.warning(f"LSTM skipped: {e}")

    save_all(models, metrics, feat_cols)
    best = min(metrics, key=lambda k: metrics[k]["rmse"])
    logger.info(f"=== Training DONE  Best={best}  RMSE={metrics[best]['rmse']} ===")
    return {"metrics": metrics, "best": best, "feature_cols": feat_cols}


if __name__ == "__main__":
    res = run_training()
    import json; print(json.dumps(res["metrics"], indent=2))