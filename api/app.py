"""api/app.py — Flask REST API."""
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from loguru import logger
from config import TARGET_CITY

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/current")
def current():
    city = request.args.get("city", TARGET_CITY)
    try:
        from feature_pipeline.fetch import fetch_all
        from config import aqi_category
        data = fetch_all(city)
        cat, col = aqi_category(data["aqi"])
        data["category"] = cat
        data["color"]    = col
        return jsonify(data)
    except Exception as e:
        logger.error(e)
        return jsonify({"error": str(e)}), 500


@app.get("/api/predict")
def predict():
    city = request.args.get("city", TARGET_CITY)
    try:
        from inference_pipeline.predict import predict_next_3_days
        return jsonify(predict_next_3_days(city=city))
    except Exception as e:
        logger.error(e)
        return jsonify({"error": str(e)}), 500


@app.get("/api/history")
def history():
    city  = request.args.get("city", TARGET_CITY)
    hours = int(request.args.get("hours", 48))
    try:
        from feature_pipeline.store import pull_latest_features
        df = pull_latest_features(city, n_hours=hours)
        return jsonify(df[["timestamp","aqi","pm25","temperature"]].to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/pipeline/run")
def run_pipeline():
    city = (request.json or {}).get("city", TARGET_CITY)
    try:
        from feature_pipeline.pipeline import run_feature_pipeline
        r = run_feature_pipeline(city=city)
        return jsonify({"status": "ok", "aqi": r["aqi"], "city": r["city"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5050))
    logger.info(f"Flask API → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
