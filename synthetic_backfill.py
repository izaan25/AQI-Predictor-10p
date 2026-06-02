"""
Standalone synthetic backfill — no API calls needed.
Generates 365 days x 24 hours of realistic AQI data directly into SQLite.
"""
import sys, math, random, sqlite3, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

city = sys.argv[1] if len(sys.argv) > 1 else "karachi"
days = int(sys.argv[2]) if len(sys.argv) > 2 else 365

ROOT     = Path(__file__).parent
DB_PATH  = ROOT / "data" / "features.db"
ROOT.joinpath("data").mkdir(exist_ok=True)

BASE_AQI  = 58
BASE_PM25 = 22
BASE_PM10 = 35

today   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
rows    = []
history = []

print(f"Generating {days * 24} hourly rows for {city}...")

for day_offset in range(days, 0, -1):
    for hour in range(24):
        dt = today - timedelta(days=day_offset) + timedelta(hours=hour)
        doy = dt.timetuple().tm_yday
        seasonal   = math.sin(2 * math.pi * doy / 365) * 12
        daily_peak = -math.cos(2 * math.pi * hour / 24) * 8
        random.seed(day_offset * 24 + hour)
        noise = random.gauss(0, 6)
        aqi   = max(10, min(300, int(BASE_AQI + seasonal + daily_peak + noise)))
        pm25  = max(1.0, round(BASE_PM25 * (aqi / BASE_AQI) + random.gauss(0,2), 1))
        pm10  = max(1.0, round(BASE_PM10 * (aqi / BASE_AQI) + random.gauss(0,4), 1))
        o3    = round(random.uniform(20,60), 1)
        no2   = round(random.uniform(10,80), 1)
        so2   = round(random.uniform(2,30),  1)
        co    = round(random.uniform(0.3,1.5),2)
        temp  = round(25 + seasonal*0.5 + random.gauss(0,3), 1)
        hum   = round(random.uniform(40,85), 1)
        wind  = round(random.uniform(1,8),   1)
        pres  = round(random.uniform(995,1015),1)

        def lag(n, key="aqi"):
            return history[-n][key] if len(history) >= n else (aqi if key=="aqi" else pm25)
        def roll(w):
            vals = [h["aqi"] for h in history[-w:]] if len(history) >= 2 else [aqi]
            return round(float(np.mean(vals)),2), round(float(np.std(vals)),2), max(vals)

        r6=roll(6); r12=roll(12); r24=roll(24)
        prev = history[-1]["aqi"] if history else aqi
        h_=dt.hour; dow=dt.weekday(); m=dt.month

        row = {
            "city": city,
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "aqi":aqi,"pm25":pm25,"pm10":pm10,"o3":o3,"no2":no2,"so2":so2,"co":co,
            "temperature":temp,"humidity":hum,"wind_speed":wind,"pressure":pres,
            "ow_temp":None,"ow_feels_like":None,"ow_humidity":None,
            "ow_pressure":None,"ow_wind_speed":None,"ow_wind_deg":None,
            "ow_cloudiness":None,"ow_visibility":None,
            "hour":h_,"day_of_week":dow,"day_of_month":dt.day,"month":m,
            "is_weekend":int(dow>=5),
            "hour_sin":round(math.sin(2*math.pi*h_/24),6),
            "hour_cos":round(math.cos(2*math.pi*h_/24),6),
            "dow_sin":round(math.sin(2*math.pi*dow/7),6),
            "dow_cos":round(math.cos(2*math.pi*dow/7),6),
            "month_sin":round(math.sin(2*math.pi*m/12),6),
            "month_cos":round(math.cos(2*math.pi*m/12),6),
            "aqi_lag_1h":lag(1),"aqi_lag_3h":lag(3),"aqi_lag_6h":lag(6),
            "aqi_lag_12h":lag(12),"aqi_lag_24h":lag(24),
            "pm25_lag_1h":lag(1,"pm25"),"pm25_lag_3h":lag(3,"pm25"),
            "pm25_lag_6h":lag(6,"pm25"),"pm25_lag_12h":lag(12,"pm25"),
            "pm25_lag_24h":lag(24,"pm25"),
            "aqi_roll_mean_6h":r6[0],"aqi_roll_std_6h":r6[1],"aqi_roll_max_6h":r6[2],
            "aqi_roll_mean_12h":r12[0],"aqi_roll_std_12h":r12[1],"aqi_roll_max_12h":r12[2],
            "aqi_roll_mean_24h":r24[0],"aqi_roll_std_24h":r24[1],"aqi_roll_max_24h":r24[2],
            "aqi_change_rate":round(aqi-prev,2),
            "aqi_change_pct":round((aqi-prev)/(prev+1e-6)*100,2),
            "pm25_pm10_ratio":round(pm25/(pm10+1e-6),4),
            "pollution_load":round(pm25+pm10+o3+no2+so2,2),
            "humidity_pm25":round(hum*pm25/100.0,2),
            "wind_aqi_ratio":round(aqi/(wind+1.0),2),
            "aqi_next_24h":None,"aqi_next_48h":None,"aqi_next_72h":None,
        }
        rows.append(row)
        history.append({"aqi":aqi,"pm25":pm25})
        if len(history) > 48:
            history = history[-48:]

df = pd.DataFrame(rows)
with sqlite3.connect(str(DB_PATH)) as conn:
    conn.execute(f"DELETE FROM features WHERE city='{city}'" )
    conn.commit()
    df.to_sql("features", conn, if_exists="append", index=False)
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]

print(f"Done. {count} rows saved to {DB_PATH}")
