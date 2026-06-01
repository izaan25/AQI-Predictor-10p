import numpy as np
import pandas as pd
from loguru import logger
from config import TARGET_CITY, aqi_category
from training_pipeline.train import load_models, FEATURE_COLS

WEIGHTS = {"rf": 0.4, "xgb": 0.3, "lstm": 0.2, "ridge": 0.1}


def _clean(X):
    return pd.DataFrame(X).fillna(0).astype(float)


def _predict_one(models, X, dampen=1.0):
    X = _clean(X.copy())
    if "aqi_change_rate" in X.columns:
        X["aqi_change_rate"] *= dampen

    preds = {}
    if "rf" in models:
        preds["rf"] = models["rf"].predict(X)[0]
    if "ridge" in models:
        Xs = models["ridge"]["scaler"].transform(X)
        preds["ridge"] = models["ridge"]["model"].predict(Xs)[0]
    if "xgb" in models:
        preds["xgb"] = models["xgb"].predict(X)[0]
    if "lstm" in models:
        Xs = models["lstm"]["scaler"].transform(X).reshape(-1, 1, X.shape[1])
        preds["lstm"] = models["lstm"]["model"].predict(Xs, verbose=0).flatten()[0]

    total_w = sum(WEIGHTS[k] for k in preds)
    val = sum(preds[k] * WEIGHTS[k] / total_w for k in preds)

    if "rf" in models:
        leaf_preds = np.array([t.predict(X)[0] for t in models["rf"].estimators_])
        std = leaf_preds.std()
    else:
        std = abs(val) * 0.10

    return max(0, min(500, float(val))), float(std)


def predict_next_3_days(city=TARGET_CITY):
    from feature_pipeline.store import pull_latest_features
    logger.info(f"=== Inference  city={city} ===")

    models, meta = load_models()
    if not models:
        raise RuntimeError("No trained models found. Run: py -3.11 -m training_pipeline.train")

    feat_cols = meta.get("feature_cols", FEATURE_COLS)
    df = pull_latest_features(city, n_hours=48)

    if df.empty:
        raise RuntimeError("No feature data. Run the feature pipeline first.")

    current_aqi = int(df["aqi"].iloc[-1])
    row = df.iloc[[-1]].copy()
    for c in feat_cols:
        if c not in row.columns:
            row[c] = 0.0
    X = _clean(row[feat_cols])

    shap_feats = []
    try:
        from training_pipeline.explain import get_shap_importance
        if "rf" in models:
            shap_feats = get_shap_importance(models["rf"], X, "rf")
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")

    predictions = []
    for horizon, dampen in [("24h", 1.0), ("48h", 0.7), ("72h", 0.4)]:
        val, std = _predict_one(models, X, dampen=dampen)
        cat, col = aqi_category(int(val))
        predictions.append({
            "horizon":    horizon,
            "aqi":        int(val),
            "aqi_low":    max(0,   int(val - std)),
            "aqi_high":   min(500, int(val + std)),
            "category":   cat,
            "color":      col,
            "confidence": round(max(0.5, 1 - std / 200), 2),
        })

    return {
        "city":        city,
        "current_aqi": current_aqi,
        "predictions": predictions,
        "shap":        shap_feats,
        "models_used": list(models.keys()),
        "metrics":     meta.get("metrics", {}),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(predict_next_3_days(), indent=2))