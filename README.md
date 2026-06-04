# 🌬️ Pearls AQI Predictor

> **100% free to run.** Real AQI data, ML models, 24-hour forecasts, SHAP explanations.

## Live Demo
https://izaan25-aqi-predictor-10p-dashboardapp-xpb189.streamlit.app/

## GitHub Repository
https://github.com/izaan25/AQI-Predictor-10p

---

## What it does

Fetches real air quality data → engineers 40+ features → trains 3 ML models (Random Forest, Ridge, XGBoost) → predicts AQI for the next 24 hours → shows everything in a live Streamlit dashboard.

---

## Free services used

| Service | What for | Cost |
|---------|----------|------|
| [AQICN](https://aqicn.org/data-platform/token/) | Real-time AQI + pollutants | **Free** (instant token) |
| [OpenWeatherMap](https://openweathermap.org/api) | Weather features | **Free** (1M calls/month) |
| SQLite (built into Python) | Local feature store | **Free** (no signup) |
| [GitHub Actions](https://github.com/features/actions) | Hourly + daily automation | **Free** (2,000 min/month) |

**No credit card required.**

---

## Requirements

- **Python 3.10, 3.11, or 3.12** — Python 3.13+ and 3.14 are NOT supported
- Windows, Mac, or Linux

---

## Step-by-step setup

### Step 1 — Get the project

```bash
# Option A: Download the ZIP and unzip it
# Option B: Clone from GitHub
git clone https://github.com/YOUR_USERNAME/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

---

### Step 2 — Check your Python versions

```bash
py --list
```

This shows all Python versions installed on your machine. You need one that is 3.10, 3.11, or 3.12.

> ⚠️ **If your default Python is 3.13 or 3.14**, you must use the explicit version flag in the next step — otherwise the venv will be broken and packages will fail.

---

### Step 3 — Create a virtual environment

Find the exact path of your Python 3.11 (or 3.10 / 3.12):

```bash
py -3.11 -c "import sys; print(sys.executable)"
```

This prints something like `C:\Python311\python.exe`. Use that full path to create the venv:

**Windows:**
```bash
C:\Python311\python.exe -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

After activation, verify the correct Python is active:

```bash
python --version
```

It must say `Python 3.11.x` (or 3.10 / 3.12). If it still shows 3.13 or 3.14, delete and recreate:

```bash
deactivate
rmdir /s /q venv        # Windows
rm -rf venv             # Mac/Linux

C:\Python311\python.exe -m venv venv
venv\Scripts\activate
```

---

### Step 4 — Install dependencies

Do **not** use `pip install -r requirements.txt` — it includes optional cloud packages that require C++ build tools and will fail. Instead run:

```bash
pip install requests loguru streamlit fastapi uvicorn scikit-learn xgboost joblib pandas numpy python-dotenv plotly
```

> **Note:** TensorFlow / LSTM is optional. The project runs fine without it — the LSTM step is skipped automatically.

---

### Step 5 — Get your free API keys

**A. AQICN token (30 seconds)**
1. Go to https://aqicn.org/data-platform/token/
2. Enter your email → click "Send Token"
3. Check your email → copy the token

**B. OpenWeather key (2 minutes)**
1. Go to https://home.openweathermap.org/users/sign_up
2. Sign up → verify email
3. Go to https://home.openweathermap.org/api_keys → copy the key
4. ⚠️ New keys take ~10 minutes to activate

---

### Step 6 — Configure environment

Open `.env` and fill in your keys:

```
AQICN_TOKEN=your_aqicn_token_here
OW_API_KEY=your_openweather_key_here
TARGET_CITY=karachi
STORAGE_MODE=local
```

---

### Step 7 — Generate training data (backfill)

```bash
python -m feature_pipeline.backfill --city karachi --days 365
```

This generates 365 days × 24 hours = 8,760 hourly records needed for model training.

> ⚠️ **Do not skip this step.** Training on the default 96 rows results in negative R² scores across all models. You need at least a few thousand rows for the lag and rolling features to be meaningful.

---

### Step 8 — Train the models

```bash
python -m training_pipeline.train
```

Trains Random Forest, Ridge, and XGBoost. Saves models to `models/latest/`.

Expected results after proper backfill:

| Model | R² | RMSE |
|-------|----|------|
| Ridge | ~0.48 | ~6.2 |
| Random Forest | ~0.45 | ~6.3 |
| XGBoost | ~0.41 | ~6.6 |

> Takes 2–5 minutes depending on your machine.

---

### Step 9 — Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser. Done! 🎉

> ⚠️ Always use `python -m streamlit` instead of just `streamlit` — the bare command is often not recognized on Windows even with the venv activated.

---

### Step 10 — Launch the API (optional)

In a second terminal, activate the venv again then run:

```bash
python -m uvicorn api.app:app --reload
```

Open http://localhost:8000

> This is optional. The dashboard works fully without it.

---

## Step 11 — GitHub Actions (automated pipeline)

The repo includes two automated workflows that run on GitHub's servers for free.

### What they do

| Workflow | Schedule | What it runs |
|----------|----------|--------------|
| Training Pipeline — Daily | 2:00 AM UTC every day | Synthetic backfill (8,760 rows) → train all 3 models |
| Feature Pipeline — Hourly | Every hour | Fetch live AQI + weather → save to DB |

### Setup (one time)

1. Push the project to GitHub
2. Go to repo → **Settings → Secrets and variables → Actions**
3. Add these **Secrets**: `AQICN_TOKEN`, `OW_API_KEY`
4. Add this **Variable**: `TARGET_CITY` = `karachi`

### Run manually

Go to **Actions → Training Pipeline — Daily → Run workflow → Run workflow**

### Important notes

- The training workflow uses `synthetic_backfill.py` (in repo root) — no API key needed for backfill, it generates data using math
- `STORAGE_MODE` must stay as `local` in the workflow files — do not change it to `hopsworks`
- Trained model artifacts are saved for 30 days under each workflow run and can be downloaded from the Actions tab
- GitHub gives 2,000 free minutes/month — well within limits for these workflows

---

## Daily usage

After the first setup, you only need:

```bash
# Activate venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Verify Python version (must be 3.11.x)
python --version

# Launch dashboard
python -m streamlit run dashboard/app.py
```

---

## Project structure

```
pearls-aqi-predictor/
│
├── .env                      ← your API keys (never commit this)
├── .env.example              ← template
├── config.py                 ← all settings
├── requirements.txt
│
├── feature_pipeline/
│   ├── fetch.py              ← AQICN + OpenWeather API calls
│   ├── engineer.py           ← 40+ features: time, lag, rolling, derived
│   ├── store.py              ← SQLite local storage
│   ├── backfill.py           ← historical data generation
│   └── pipeline.py           ← hourly orchestrator
│
├── training_pipeline/
│   ├── train.py              ← RF, Ridge, XGBoost (+ optional LSTM)
│   └── explain.py            ← SHAP feature importance
│
├── inference_pipeline/
│   └── predict.py            ← 24h ensemble forecast
│
├── api/
│   └── app.py                ← FastAPI REST API (port 8000)
│
├── dashboard/
│   └── app.py                ← Streamlit dashboard (port 8501)
│
├── models/latest/            ← trained model files
├── data/features.db          ← local SQLite store
│
└── .github/workflows/
    ├── feature_pipeline.yml  ← hourly (GitHub Actions)
    └── training_pipeline.yml ← daily  (GitHub Actions)
```

---

## Common issues

| Problem | Fix |
|---------|-----|
| `No module named 'requests'` | `pip install requests` |
| `No module named 'plotly'` | `pip install plotly` |
| `No module named 'loguru'` | `pip install loguru` |
| `No module named 'streamlit'` | Venv is using wrong Python — recreate with the full path method in Step 3 |
| `streamlit is not recognized` | Use `python -m streamlit run dashboard/app.py` |
| `uvicorn is not recognized` | Use `python -m uvicorn api.app:app --reload` |
| `Failed building wheel for twofish` | Use the manual `pip install` command in Step 4, not `requirements.txt` |
| `tensorflow` won't install | Skip it — LSTM is optional, the other 3 models work fine |
| `Only N rows — need at least 30` | Run Step 7 (backfill) first |
| `Train models to see SHAP importances` | Run Step 8 (training) first |
| `Predictions unavailable: No module named tensorflow` | Edit `training_pipeline/train.py` — wrap the LSTM load block in `try/except Exception: pass` |
| Negative R² scores | Not enough training data — run backfill with `--days 365` |
| Python version not changing in venv | Use full Python path: `C:\Python311\python.exe -m venv venv` (see Step 3) |
| Python 3.13 / 3.14 errors | These versions are unsupported — install Python 3.11 from python.org |

---

## Supported city

`karachi`

Add more in `config.py → CITY_COORDS`.

---

## AQI scale

| Range | Category |
|-------|----------|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301–500 | Hazardous |

---

MIT License — free to use and modify.
