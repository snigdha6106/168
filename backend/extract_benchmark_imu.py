import pandas as pd
import json

df = pd.read_csv("data/real_route_processed.csv")
df = df.dropna(subset=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'gnss_lat', 'gnss_lon'])

records = []
for idx, row in df.head(1000).iterrows():
    records.append({
        "lat": row['gnss_lat'],
        "lon": row['gnss_lon'],
        "imu": [
            row['acc_x'], row['acc_y'], row['acc_z'],
            row['gyro_x'], row['gyro_y'], row['gyro_z']
        ]
    })

with open("../frontend/assets/benchmark_imu_stream.json", "w") as f:
    json.dump(records, f)

print("Exported 1000 IMU frames to frontend/assets/benchmark_imu_stream.json")
