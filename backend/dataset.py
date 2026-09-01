import numpy as np
import pandas as pd
import os

def generate_synthetic_route(num_points=2000, start_lat=28.6139, start_lon=77.2090, dt=0.1):
    """
    Generates a synthetic vehicle trajectory simulating consumer IMU noise and GNSS dropouts.
    start_lat/lon default to New Delhi.
    """
    time = np.arange(0, num_points * dt, dt)
    
    # Base velocity (m/s) -> roughly 15 m/s (54 km/h)
    velocity = 15.0 + 2.0 * np.sin(0.02 * time)
    
    # Heading (radians) -> slowly changing (some turns)
    heading_changes = 0.005 * np.random.randn(num_points)
    # Simulate a perfectly straight tunnel between 600 and 1200
    heading_changes[600:1200] = 0.0
    heading = np.cumsum(heading_changes) + (np.pi/4) # Start moving North-East
    
    # True positions
    lat, lon = np.zeros(num_points), np.zeros(num_points)
    lat[0], lon[0] = start_lat, start_lon
    
    # Approximate meters to degrees conversion
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(start_lat))
    
    # Generate true path and ideal IMU (acceleration, gyro Z)
    acc_x = np.zeros(num_points) # forward acceleration
    gyro_z = np.zeros(num_points) # yaw rate
    
    for i in range(1, num_points):
        # Calculate ideal IMU
        acc_x[i] = (velocity[i] - velocity[i-1]) / dt
        gyro_z[i] = (heading[i] - heading[i-1]) / dt
        
        # Update position
        v_y = velocity[i] * np.cos(heading[i]) # North
        v_x = velocity[i] * np.sin(heading[i]) # East
        
        lat[i] = lat[i-1] + (v_y * dt) / m_per_deg_lat
        lon[i] = lon[i-1] + (v_x * dt) / m_per_deg_lon
        
    # Add noise to IMU to simulate consumer grade smartphone sensors
    acc_x_noisy = acc_x + np.random.normal(0, 0.3, num_points) + 0.1 # noise + bias
    gyro_z_noisy = gyro_z + np.random.normal(0, 0.02, num_points) + 0.002 # noise + smaller realistic bias
    
    # Add 3D axes (y, z) which we assume mostly zero (gravity on z handled/filtered in real app)
    acc_y_noisy = np.random.normal(0, 0.5, num_points)
    acc_z_noisy = np.random.normal(9.81, 0.5, num_points)
    gyro_x_noisy = np.random.normal(0, 0.05, num_points)
    gyro_y_noisy = np.random.normal(0, 0.05, num_points)
    
    # Generate GNSS data with some dropouts
    gnss_lat = lat + np.random.normal(0, 3.0 / m_per_deg_lat, num_points) # 3m noise
    gnss_lon = lon + np.random.normal(0, 3.0 / m_per_deg_lon, num_points) # 3m noise
    gnss_status = np.ones(num_points)
    
    # Simulate a tunnel or dense urban canyon blackout
    dropout_start = 600
    dropout_end = 1200
    gnss_status[dropout_start:dropout_end] = 0
    
    df = pd.DataFrame({
        'timestamp': time,
        'true_lat': lat,
        'true_lon': lon,
        'true_velocity': velocity,
        'true_heading': heading,
        'gnss_lat': gnss_lat,
        'gnss_lon': gnss_lon,
        'gnss_status': gnss_status,
        'acc_x': acc_x_noisy,
        'acc_y': acc_y_noisy,
        'acc_z': acc_z_noisy,
        'gyro_x': gyro_x_noisy,
        'gyro_y': gyro_y_noisy,
        'gyro_z': gyro_z_noisy
    })
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_route()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_route.csv", index=False)
    print("Synthetic dataset generated successfully at data/synthetic_route.csv.")
