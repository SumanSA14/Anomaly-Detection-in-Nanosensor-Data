import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

LIVE_DELAY = 1  
ANOMALY_INTERVAL = 15  
CSV_PATH = "dataset_final.csv"
sensor_cols = ['Temperature', 'Humidity', 'Air Quality', 'Light', 'Loudness']

df = pd.read_csv(CSV_PATH)
df['Timestamp'] = pd.to_datetime(df['Time'], unit='s')
df = df[sensor_cols].copy()
df = df.reset_index(drop=True)

print("\nSimulating data with artificial anomalies every 15 seconds...\n")
print("Timestamp\t\t\tStatus\t\tData")

start_time = datetime.now()
anomaly_toggle_time = start_time
is_anomaly = False

for i in range(len(df)):
    now = datetime.now()

    
    if (now - anomaly_toggle_time).total_seconds() >= ANOMALY_INTERVAL:
        is_anomaly = not is_anomaly
        anomaly_toggle_time = now

    data = df.iloc[i].copy()

    if is_anomaly:
        
        data['Temperature'] += np.random.uniform(5, 10)
        data['Humidity'] += np.random.uniform(10, 20)
        data['Light'] += np.random.uniform(100, 300)
        status = "ANOMALY"
    else:
        status = "NORMAL"

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    values_str = " | ".join(f"{val:.2f}" for val in data.values)
    print(f"{timestamp} \t {status:<8} \t {values_str}")

    time.sleep(LIVE_DELAY)
