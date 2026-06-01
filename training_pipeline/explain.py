"""training_pipeline/explain.py — SHAP feature importance."""
import shap
import numpy as np
import pandas as pd
from loguru import logger


def get_shap_importance(model, X: pd.DataFrame, model_type: str = "rf", top_n: int = 10) -> list[dict]:
    """Return top N features by mean |SHAP| value."""
    sample = X.iloc[:min(200, len(X))]
    try:
        if model_type in ("rf", "xgb"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(sample)
        elif model_type == "ridge":
            actual = model["model"] if isinstance(model, dict) else model
            sc     = model["scaler"] if isinstance(model, dict) else None
            Xs = pd.DataFrame(sc.transform(sample), columns=sample.columns) if sc else sample
            explainer = shap.LinearExplainer(actual, Xs)
            sv = explainer.shap_values(Xs)
        else:
            return []

        mean_abs  = np.abs(sv).mean(axis=0)
        mean_shap = sv.mean(axis=0)
        cols = list(sample.columns)

        top = sorted(
            [{"feature": cols[i], "importance": round(float(mean_abs[i]),4),
              "direction": "positive" if mean_shap[i] > 0 else "negative"}
             for i in range(len(cols))],
            key=lambda x: x["importance"], reverse=True
        )
        return top[:top_n]
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        # Fallback: use feature importances from tree models
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            return sorted(
                [{"feature": c, "importance": round(float(fi[i]),4), "direction": "positive"}
                 for i, c in enumerate(sample.columns)],
                key=lambda x: x["importance"], reverse=True
            )[:top_n]
        return []
