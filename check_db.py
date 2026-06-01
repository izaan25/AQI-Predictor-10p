import sqlite3
conn = sqlite3.connect('data/features.db')
count = conn.execute('SELECT COUNT(*) FROM features').fetchone()
print('Total rows:', count[0])
sample = conn.execute('SELECT city, timestamp, aqi, pm25 FROM features LIMIT 5').fetchall()
for row in sample:
    print(row)
conn.close()