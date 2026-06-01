import pandas as pd
import sqlite3
import sys
sys.path.insert(0, '.')
from config import DB_PATH

df = pd.DataFrame([{'city':'karachi','aqi':100,'pm25':50.0}])
print('DB path:', DB_PATH)
print('Writing test row...')
with sqlite3.connect(str(DB_PATH)) as conn:
    df.to_sql('features', conn, if_exists='append', index=False)
    conn.commit()
    count = conn.execute('SELECT COUNT(*) FROM features').fetchone()
    print('Rows in DB:', count)
print('SUCCESS!')