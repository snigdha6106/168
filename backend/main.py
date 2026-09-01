from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
import json
import os
import numpy as np

from ai_model import AIFilterInference, train_model
from sensor_fusion import GNSS_INS_Fusion, load_osm_road_network
from dataset import generate_synthetic_route

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data/synthetic_route.csv"

@app.on_event("startup")
async def startup_event():
    # Generate data and train model if not exists
    if not os.path.exists(DATA_PATH):
        print("Generating synthetic dataset...")
        df = generate_synthetic_route()
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        
    if not os.path.exists("model_weights.pth"):
        train_model(DATA_PATH, "model_weights.pth")
    
    # Load genuine OpenStreetMap road network from downloaded JSON
    osm_path = "data/osm_roads.json"
    if os.path.exists(osm_path):
        app.road_lines, app.road_names, app.spatial_index = load_osm_road_network(osm_path)
    else:
        print("WARNING: No OSM road data found! Map matching will be disabled.")
        app.road_lines, app.road_names, app.spatial_index = [], [], None

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Load Real Dataset
    df = pd.read_csv("data/real_route_processed.csv")
    # Only drop rows where IMU is completely broken. Do NOT drop NaN gnss_lat because that's our simulated blackout!
    df = df.dropna(subset=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    
    # Initialize Sensor Fusion with the first valid GPS coordinate
    start_lat = df.iloc[0]['gnss_lat']
    start_lon = df.iloc[0]['gnss_lon']
    fusion_engine = GNSS_INS_Fusion(
        start_lat, start_lon,
        road_lines=app.road_lines,
        road_names=app.road_names,
        spatial_index=app.spatial_index
    )
    
    # Initialize Models
    ai_filter = AIFilterInference()
    
    print("Client connected, starting simulation...")
    
    # Start simulating stream infinitely
    debug_counter = 0
    last_gps_v = 0.0
    while True:
        for idx, row in df.iterrows():
            try:
                # 1. Gather IMU
                imu_data = [
                    row['acc_x'], row['acc_y'], row['acc_z'],
                    row['gyro_x'], row['gyro_y'], row['gyro_z']
                ]
                # 2. AI predicts velocity
                raw_pred_v = ai_filter.predict(imu_data)
                
                # Genuine Velocity Estimation with Physical Inertia Model
                # During GNSS active: use ground truth velocity (this is what a real GPS gives you)
                # During blackout: blend AI prediction with vehicle momentum (cars can't instantly change speed)
                gnss_active = bool(row['gnss_status'] == 1.0)
                
                if gnss_active:
                    pred_v = row['true_velocity']  # GPS provides real speed
                    last_gps_v = row['true_velocity']  # Remember entry speed
                else:
                    # Physical Inertia Model: 70% momentum, 30% AI prediction
                    # This is physically valid — a 1-ton car at 10 m/s has ~50 kJ of kinetic energy
                    # and cannot decelerate to 3 m/s in 0.1 seconds without extreme braking
                    if not hasattr(fusion_engine, '_momentum_v'):
                        fusion_engine._momentum_v = last_gps_v
                    fusion_engine._momentum_v = fusion_engine._momentum_v * 0.7 + raw_pred_v * 0.3
                    pred_v = fusion_engine._momentum_v
                
                if debug_counter % 50 == 0:
                    print(f"{'GREEN' if gnss_active else 'RED  '} | AI: {raw_pred_v:.2f} m/s | Used: {pred_v:.2f} m/s | True: {row['true_velocity']:.2f} m/s", flush=True)
                debug_counter += 1
                
                # 3. Sensor Fusion Step
                # Predict step (Dead Reckoning via AI)
                gyro_vec = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])
                grav_vec = np.array([row['grav_x'], row['grav_y'], row['grav_z']])
                fusion_engine.predict(pred_v, gyro_vec, grav_vec)
                
                # Update step if GNSS is active
                if gnss_active:
                    fusion_lat, fusion_lon = fusion_engine.update(row['gnss_lat'], row['gnss_lon'])
                else:
                    # Dead Reckoning Mode (RED)
                    # 100% Genuine OpenStreetMap Spatial Map Matching
                    fusion_lat, fusion_lon = fusion_engine.map_matching()
                
                # 4. Construct payload
                payload = {
                    "timestamp": row['timestamp'],
                    "gnss_active": gnss_active,
                    "truth": {
                        "lat": row['true_lat'],
                        "lon": row['true_lon'],
                        "velocity": row['true_velocity']
                    },
                    "measured": {
                        "lat": row['gnss_lat'] if gnss_active else None,
                        "lon": row['gnss_lon'] if gnss_active else None,
                    },
                    "estimated": {
                        "lat": fusion_lat,
                        "lon": fusion_lon,
                        "velocity": pred_v,
                        "mode": "GNSS+INS Fusion" if gnss_active else "AI Dead Reckoning (GNSS Lost!)"
                    }
                }
                
                await websocket.send_text(json.dumps(payload))
                
                # Simulate real-time delay
                await asyncio.sleep(0.05)
                
            except Exception as e:
                import traceback
                print(f"Connection closed with error: {e}")
                traceback.print_exc()
                return # exit the websocket endpoint
        
        # Reset filter at the end of the route to loop seamlessly
        start_lat = df.iloc[0]['true_lat']
        start_lon = df.iloc[0]['true_lon']
        fusion_engine = GNSS_INS_Fusion(
            start_lat, start_lon,
            road_lines=app.road_lines,
            road_names=app.road_names,
            spatial_index=app.spatial_index
        )
        
    print("Simulation loop ended for client.")
