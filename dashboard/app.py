"""dashboard/app.py — Streamlit AQI dashboard (light/dark mode compatible)."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TARGET_CITY, CITY_COORDS, aqi_category
from feature_pipeline.fetch import fetch_all
from feature_pipeline.store import pull_latest_features

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌬️", layout="wide")

# ── Theme-aware CSS (works on both light and dark mode) ───────
st.markdown("""
<style>
/* Remove forced dark background — let Streamlit handle theming */
.stMetric {
    border-radius: 8px;
    padding: 8px;
    border: 1px solid rgba(128,128,128,0.2);
}
.stMetric label {
    font-size: 0.8rem !important;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌬️ Pearls AQI")
    city = st.selectbox("City", list(CITY_COORDS.keys()), index=0)
    if st.button("🔄 Refresh"):
        st.cache_data.clear(); st.rerun()
    st.markdown("---")
    st.markdown("**Storage**")
    from config import STORAGE_MODE, DB_PATH
    st.markdown(f"Mode: `{STORAGE_MODE}`")
    if STORAGE_MODE == "local":
        exists = DB_PATH.exists()
        st.markdown(f"DB: {'✅ exists' if exists else '⚠️ run pipeline first'}")
    st.markdown("---")
    st.markdown("**Links**")
    st.markdown("• [AQICN token](https://aqicn.org/data-platform/token/)")
    st.markdown("• [OpenWeather key](https://openweathermap.org/api)")
    st.markdown("• [Hopsworks free](https://app.hopsworks.ai)")

# ── Live data ─────────────────────────────────────────────────
@st.cache_data(ttl=1)
def get_current(city):
    try:
        d = fetch_all(city)
        cat, col = aqi_category(d["aqi"])
        d["category"] = cat; d["color"] = col
        return d
    except Exception as e:
        st.error(f"API fetch failed: {e}")
        return {}

@st.cache_data(ttl=600)
def get_predictions(city):
    try:
        from inference_pipeline.predict import predict_next_3_days
        return predict_next_3_days(city=city)
    except Exception as e:
        return {"error": str(e), "predictions": [], "shap": [], "metrics": {}}

@st.cache_data(ttl=600)
def get_history(city):
    try:
        return pull_latest_features(city, n_hours=48)
    except Exception:
        return pd.DataFrame()

current = get_current(city)
preds   = get_predictions(city)
history = get_history(city)

# ── Header ────────────────────────────────────────────────────
st.title(f"🌬️ Pearls AQI Predictor — {city.capitalize()}")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  Ensemble ML (RF + LSTM + XGBoost + Ridge)")

if current.get("aqi", 0) > 150:
    st.error(f"⚠️ Hazardous AQI Alert: {current.get('aqi')} — {current.get('category')}. Stay indoors.")

# ── Current row ───────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Current AQI",  current.get("aqi",        "–"))
c2.metric("Category",     current.get("category",   "–"))
c3.metric("PM2.5 µg/m³",  current.get("pm25",       "–"))
c4.metric("PM10 µg/m³",   current.get("pm10",       "–"))
c5.metric("Temp °C",      current.get("temperature", current.get("ow_temp","–")))
c6.metric("Humidity %",   current.get("humidity",    current.get("ow_humidity","–")))

st.divider()

# ── Forecast ──────────────────────────────────────────────────
st.subheader("📅 72-Hour Forecast")
if preds.get("error"):
    st.warning(f"Predictions unavailable: {preds['error']}\n\nRun: `python -m training_pipeline.train`")
else:
    cols = st.columns(3)
    labels = ["Today +24h", "Tomorrow +48h", "Day 3 +72h"]
    for col, pred, lbl in zip(cols, preds.get("predictions",[]), labels):
        with col:
            st.markdown(f"**{lbl}**")
            delta = pred["aqi"] - current.get("aqi", pred["aqi"])
            st.metric(pred["category"], pred["aqi"], delta=delta, delta_color="inverse")
            st.progress(min(1.0, pred["aqi"]/300))
            st.caption(f"Confidence: {pred['confidence']*100:.0f}%  ·  Range: {pred['aqi_low']}–{pred['aqi_high']}")

st.divider()

# ── Timeline chart ────────────────────────────────────────────
st.subheader("📈 AQI Timeline")
if not history.empty and "timestamp" in history.columns and "aqi" in history.columns:
    hist_ts  = pd.to_datetime(history["timestamp"])
    hist_aqi = history["aqi"]
    now = pd.Timestamp.now()
    pred_ts  = [now + pd.Timedelta(hours=24*i) for i in [1,2,3]]
    pred_aqi = [p["aqi"] for p in preds.get("predictions",[])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_ts, y=hist_aqi, mode="lines+markers",
        name="Historical", line=dict(color="#38bdf8", width=2), marker=dict(size=4)))
    if pred_aqi:
        fig.add_trace(go.Scatter(
            x=[hist_ts.iloc[-1]] + pred_ts, y=[hist_aqi.iloc[-1]] + pred_aqi,
            mode="lines+markers", name="Forecast",
            line=dict(color="#a78bfa", width=2, dash="dash"), marker=dict(size=6)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)", showgrid=True),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)", showgrid=True, title="AQI"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No history yet. Run the feature pipeline first.")

st.divider()

# ── Pollutants ────────────────────────────────────────────────
st.subheader("🧪 Pollutants")
pc = st.columns(6)
for col, (name, key, unit, mx) in zip(pc, [
    ("PM2.5","pm25","µg/m³",150), ("PM10","pm10","µg/m³",250),
    ("O₃","o3","ppb",100), ("NO₂","no2","ppb",150),
    ("SO₂","so2","ppb",75), ("CO","co","ppm",4),
]):
    val = current.get(key)
    with col:
        st.metric(name, f"{val or '–'} {unit}")
        if val: st.progress(min(1.0, float(val)/mx))

st.divider()

# ── SHAP ──────────────────────────────────────────────────────
st.subheader("🧠 SHAP Feature Importance")
shap_data = preds.get("shap", [])
if shap_data:
    df_shap = pd.DataFrame(shap_data)
    fig2 = px.bar(df_shap.head(10), x="importance", y="feature", orientation="h",
                  color="direction",
                  color_discrete_map={"positive":"#fb7185","negative":"#4ade80"})
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        showlegend=True,
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Train models to see SHAP importances.")

st.divider()

# ── Model metrics ─────────────────────────────────────────────
st.subheader("📊 Model Performance")
metrics = preds.get("metrics", {})
if metrics:
    rows = [{"Model": k.upper(), "RMSE": v["rmse"], "MAE": v["mae"], "R²": v["r2"]}
            for k,v in metrics.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Run training pipeline to see metrics.")

st.caption("Pearls AQI Predictor · AQICN + OpenWeather · RF + LSTM + XGBoost + Ridge · SHAP")
