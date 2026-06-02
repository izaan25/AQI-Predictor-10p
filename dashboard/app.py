"""dashboard/app.py — Pearls AQI Predictor — Enhanced Dashboard"""
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

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Header */
.aqi-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.aqi-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.aqi-hero h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem !important;
    color: #f1f5f9 !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.5px;
}
.aqi-hero .subtitle {
    color: #64748b;
    font-size: 0.85rem;
    font-family: 'Space Mono', monospace;
}
.aqi-hero .live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.3);
    color: #4ade80;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 12px;
}
.live-dot {
    width: 7px; height: 7px;
    background: #4ade80;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.8); }
}

/* AQI Big number card */
.aqi-big-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.aqi-big-card .aqi-num {
    font-family: 'Space Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
    margin: 8px 0 4px;
}
.aqi-big-card .aqi-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'Space Mono', monospace;
}
.aqi-big-card .aqi-cat {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 6px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f172a, #1a2744);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 12px;
    padding: 16px 18px;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover {
    border-color: rgba(56,189,248,0.4);
    transform: translateY(-2px);
}
.metric-card .mc-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'Space Mono', monospace;
    margin-bottom: 6px;
}
.metric-card .mc-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    font-family: 'Space Mono', monospace;
    line-height: 1.2;
}
.metric-card .mc-unit {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 2px;
}

/* Forecast cards */
.forecast-card {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(100,116,139,0.25);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: all 0.25s;
}
.forecast-card:hover {
    border-color: rgba(167,139,250,0.5);
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(167,139,250,0.1);
}
.forecast-card .fc-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'Space Mono', monospace;
}
.forecast-card .fc-aqi {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 8px 0 4px;
}
.forecast-card .fc-cat {
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 10px;
}
.forecast-card .fc-range {
    font-size: 0.72rem;
    color: #64748b;
    font-family: 'Space Mono', monospace;
}
.forecast-card .fc-conf {
    font-size: 0.72rem;
    color: #4ade80;
    font-family: 'Space Mono', monospace;
    margin-top: 4px;
}

/* Pollutant bar */
.poll-bar-wrap {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.poll-name { font-size:0.78rem; color:#94a3b8; font-family:'Space Mono',monospace; margin-bottom:4px; }
.poll-val  { font-size:1.2rem; font-weight:700; color:#e2e8f0; font-family:'Space Mono',monospace; }
.poll-track { background:rgba(100,116,139,0.15); border-radius:4px; height:5px; margin-top:8px; }
.poll-fill  { height:5px; border-radius:4px; transition:width 0.5s ease; }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #38bdf8;
    margin: 28px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(56,189,248,0.3), transparent);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080e1a !important;
    border-right: 1px solid rgba(56,189,248,0.1) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #64748b !important; }

/* Hide default streamlit metric styling */
[data-testid="stMetric"] {
    background: transparent !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Data fetching ─────────────────────────────────────────────
from config import STORAGE_MODE, DB_PATH

@st.cache_data(ttl=60)
def get_current(city):
    try:
        d = fetch_all(city)
        cat, col = aqi_category(d["aqi"])
        d["category"] = cat; d["color"] = col
        return d
    except Exception as e:
        return {"_error": str(e)}

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

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌬️ Pearls AQI")
    st.markdown("---")
    city = st.selectbox("**Select City**", list(CITY_COORDS.keys()), index=0)

    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.markdown("---")
    st.markdown("**⚙️ System Status**")
    st.markdown(f"Storage: `{STORAGE_MODE}`")
    if STORAGE_MODE == "local":
        st.markdown(f"DB: {'✅ Ready' if DB_PATH.exists() else '⚠️ Missing'}")
    st.markdown(f"Last update: `{datetime.now().strftime('%H:%M:%S')}`")

    st.markdown("---")
    st.markdown("**🔗 Resources**")
    st.markdown("• [AQICN Token](https://aqicn.org/data-platform/token/)")
    st.markdown("• [OpenWeather API](https://openweathermap.org/api)")
    st.markdown("• [GitHub Repo](https://github.com/izaan25/AQI-Predictor-10p)")

if auto_refresh:
    import time
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()

# ── Load data ─────────────────────────────────────────────────
current = get_current(city)
preds   = get_predictions(city)
history = get_history(city)

aqi_val  = current.get("aqi", 0)
aqi_cat  = current.get("category", "Unknown")
aqi_col  = current.get("color", "#94a3b8")

# AQI color mapping
def aqi_color(val):
    if val <= 50:   return "#4ade80"
    if val <= 100:  return "#facc15"
    if val <= 150:  return "#fb923c"
    if val <= 200:  return "#f87171"
    if val <= 300:  return "#c084fc"
    return "#fb7185"

# ── Hero Header ───────────────────────────────────────────────
if current.get("_error"):
    st.error(f"⚠️ API fetch failed: {current['_error']}")

st.markdown(f"""
<div class="aqi-hero">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
    <div>
      <h1>🌬️ Pearls AQI Predictor
        <span class="live-badge"><span class="live-dot"></span>LIVE</span>
      </h1>
      <div class="subtitle">
        {city.upper()} &nbsp;·&nbsp; {datetime.now().strftime('%A, %d %B %Y  %H:%M')}
        &nbsp;·&nbsp; Ensemble ML: RF + XGBoost + Ridge
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:2px;">Current AQI</div>
      <div style="font-family:'Space Mono',monospace; font-size:3rem; font-weight:700; color:{aqi_color(aqi_val)}; line-height:1.1;">{aqi_val if aqi_val else '—'}</div>
      <div style="font-size:0.95rem; font-weight:600; color:{aqi_color(aqi_val)};">{aqi_cat}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if aqi_val > 150:
    st.error(f"⚠️ **Air Quality Alert** — AQI {aqi_val} ({aqi_cat}). Limit outdoor activities.")
elif aqi_val > 100:
    st.warning(f"⚠️ **Moderate Pollution** — AQI {aqi_val}. Sensitive groups should take precautions.")

# ── Current Conditions ────────────────────────────────────────
st.markdown('<div class="section-header">📡 Current Conditions</div>', unsafe_allow_html=True)

cols = st.columns(5)
metrics = [
    ("PM2.5", current.get("pm25","–"), "µg/m³"),
    ("PM10",  current.get("pm10","–"), "µg/m³"),
    ("Temperature", current.get("temperature", current.get("ow_temp","–")), "°C"),
    ("Humidity", current.get("humidity", current.get("ow_humidity","–")), "%"),
    ("Wind Speed", current.get("wind_speed", current.get("ow_wind_speed","–")), "m/s"),
]
for col, (label, val, unit) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div class="mc-label">{label}</div>
      <div class="mc-value">{val}</div>
      <div class="mc-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)

# ── 72-Hour Forecast ──────────────────────────────────────────
st.markdown('<div class="section-header">📅 72-Hour Forecast</div>', unsafe_allow_html=True)

if preds.get("error"):
    st.warning(f"Predictions unavailable: {preds['error']}\n\nRun: `python -m training_pipeline.train`")
elif preds.get("predictions"):
    labels = ["Today +24h", "Tomorrow +48h", "Day 3 +72h"]
    icons  = ["🌅", "🌤️", "🌍"]
    fcols  = st.columns(3)
    for col, pred, lbl, icon in zip(fcols, preds["predictions"], labels, icons):
        delta = pred["aqi"] - aqi_val if aqi_val else 0
        delta_str = f"▲ +{delta}" if delta > 0 else f"▼ {delta}"
        delta_col = "#f87171" if delta > 0 else "#4ade80"
        col.markdown(f"""
        <div class="forecast-card">
          <div class="fc-label">{icon} {lbl}</div>
          <div class="fc-aqi" style="color:{aqi_color(pred['aqi'])};">{pred['aqi']}</div>
          <div class="fc-cat" style="color:{aqi_color(pred['aqi'])};">{pred['category']}</div>
          <div style="height:4px; background:rgba(100,116,139,0.15); border-radius:4px; margin:10px 0;">
            <div style="width:{min(100, pred['aqi']/3):.0f}%; height:4px; background:{aqi_color(pred['aqi'])}; border-radius:4px;"></div>
          </div>
          <div class="fc-range">Range: {pred['aqi_low']}–{pred['aqi_high']}</div>
          <div class="fc-conf">Confidence: {pred['confidence']*100:.0f}%</div>
          <div style="font-size:0.72rem; color:{delta_col}; font-family:'Space Mono',monospace; margin-top:4px;">{delta_str} vs now</div>
        </div>""", unsafe_allow_html=True)

# ── AQI Timeline ──────────────────────────────────────────────
st.markdown('<div class="section-header">📈 AQI Timeline</div>', unsafe_allow_html=True)

if not history.empty and "timestamp" in history.columns:
    hist_ts  = pd.to_datetime(history["timestamp"])
    hist_aqi = history["aqi"]
    now_ts   = pd.Timestamp.now()
    pred_ts  = [now_ts + pd.Timedelta(hours=24*i) for i in [1,2,3]]
    pred_aqi = [p["aqi"] for p in preds.get("predictions",[])]

    fig = go.Figure()

    # AQI zone bands
    for lo, hi, c in [(0,50,"#4ade80"),(51,100,"#facc15"),(101,150,"#fb923c"),(151,200,"#f87171")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=c, opacity=0.04, line_width=0)

    fig.add_trace(go.Scatter(
        x=hist_ts, y=hist_aqi, mode="lines",
        name="Historical AQI",
        line=dict(color="#38bdf8", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.06)",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y}<extra></extra>"
    ))

    if pred_aqi:
        fig.add_trace(go.Scatter(
            x=[hist_ts.iloc[-1]] + pred_ts,
            y=[hist_aqi.iloc[-1]] + pred_aqi,
            mode="lines+markers", name="Forecast",
            line=dict(color="#a78bfa", width=2, dash="dot"),
            marker=dict(size=8, color="#a78bfa", symbol="diamond"),
            hovertemplate="<b>Forecast %{x|%b %d}</b><br>AQI: %{y}<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            gridcolor="rgba(100,116,139,0.15)",
            showgrid=True, zeroline=False,
            tickfont=dict(family="Space Mono", size=10, color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="rgba(100,116,139,0.15)",
            tickfont=dict(family="Space Mono", size=10, color="#64748b"),
            title=dict(text="AQI", font=dict(color="#64748b", size=11)),
            autorange="reversed"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono", size=10, color="#94a3b8"),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No history yet — run the feature pipeline first.")

# ── Pollutants ────────────────────────────────────────────────
st.markdown('<div class="section-header">🧪 Pollutant Breakdown</div>', unsafe_allow_html=True)

pollutants = [
    ("PM2.5", "pm25", "µg/m³", 150, "#38bdf8"),
    ("PM10",  "pm10", "µg/m³", 250, "#818cf8"),
    ("O₃ Ozone",   "o3",  "ppb",   100, "#4ade80"),
    ("NO₂",  "no2", "ppb",   150, "#facc15"),
    ("SO₂",  "so2", "ppb",   75,  "#fb923c"),
    ("CO",   "co",  "ppm",   4,   "#f87171"),
]

pc = st.columns(6)
for col, (name, key, unit, mx, clr) in zip(pc, pollutants):
    val = current.get(key)
    pct = min(100, float(val)/mx*100) if val else 0
    col.markdown(f"""
    <div class="poll-bar-wrap">
      <div class="poll-name">{name}</div>
      <div class="poll-val">{val if val else '–'}<span style="font-size:0.65rem;color:#475569;margin-left:4px;">{unit}</span></div>
      <div class="poll-track">
        <div class="poll-fill" style="width:{pct:.0f}%;background:{clr};"></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── SHAP ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧠 Feature Importance (SHAP)</div>', unsafe_allow_html=True)

shap_data = preds.get("shap", [])
if shap_data:
    df_shap = pd.DataFrame(shap_data).head(12)
    fig2 = go.Figure(go.Bar(
        x=df_shap["importance"],
        y=df_shap["feature"],
        orientation="h",
        marker=dict(
            color=["#fb7185" if d=="positive" else "#4ade80" for d in df_shap["direction"]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            gridcolor="rgba(100,116,139,0.15)",
            tickfont=dict(family="Space Mono", size=10, color="#64748b"),
            title="SHAP Importance", titlefont=dict(color="#64748b")
        ),
        yaxis=dict(
            gridcolor="rgba(100,116,139,0.15)",
            tickfont=dict(family="Space Mono", size=10, color="#94a3b8"),
            autorange="reversed"
        ),
    )
    l, r = st.columns([3,1])
    with l:
        st.plotly_chart(fig2, use_container_width=True)
    with r:
        st.markdown("""
        <div style="padding:16px; background:rgba(15,23,42,0.8); border:1px solid rgba(100,116,139,0.2); border-radius:10px; margin-top:8px;">
          <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;">Legend</div>
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <div style="width:12px;height:12px;background:#fb7185;border-radius:2px;"></div>
            <span style="font-size:0.8rem; color:#94a3b8;">Increases AQI</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:12px;height:12px;background:#4ade80;border-radius:2px;"></div>
            <span style="font-size:0.8rem; color:#94a3b8;">Decreases AQI</span>
          </div>
        </div>""", unsafe_allow_html=True)
else:
    st.info("Train models to see SHAP importances.")

# ── Model Performance ─────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)

model_metrics = preds.get("metrics", {})
if model_metrics:
    mc = st.columns(len(model_metrics))
    colors = ["#38bdf8", "#a78bfa", "#4ade80"]
    for col, ((k, v), clr) in zip(mc, zip(model_metrics.items(), colors)):
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);
                    border:1px solid {clr}33; border-top:3px solid {clr};
                    border-radius:12px; padding:20px; text-align:center;">
          <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                      color:{clr}; text-transform:uppercase; letter-spacing:2px;
                      margin-bottom:12px;">{k.upper()}</div>
          <div style="margin-bottom:8px;">
            <div style="font-size:0.65rem; color:#64748b; font-family:'Space Mono',monospace;">R²</div>
            <div style="font-size:1.8rem; font-weight:700; color:#e2e8f0;
                        font-family:'Space Mono',monospace;">{v['r2']:.3f}</div>
          </div>
          <div style="display:flex; justify-content:space-around; margin-top:8px;">
            <div style="text-align:center;">
              <div style="font-size:0.65rem; color:#64748b; font-family:'Space Mono',monospace;">RMSE</div>
              <div style="font-size:1rem; font-weight:600; color:#94a3b8;
                          font-family:'Space Mono',monospace;">{v['rmse']}</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:0.65rem; color:#64748b; font-family:'Space Mono',monospace;">MAE</div>
              <div style="font-size:1rem; font-weight:600; color:#94a3b8;
                          font-family:'Space Mono',monospace;">{v['mae']}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
else:
    st.info("Run training pipeline to see model metrics.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:20px 0; border-top:1px solid rgba(100,116,139,0.15);">
  <span style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#334155;">
    Pearls AQI Predictor &nbsp;·&nbsp; AQICN + OpenWeather &nbsp;·&nbsp;
    RF + XGBoost + Ridge + SHAP &nbsp;·&nbsp; 10Pearls Internship
  </span>
</div>
""", unsafe_allow_html=True)
