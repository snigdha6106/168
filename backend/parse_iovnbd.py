import pandas as pd
import numpy as np
import os

def parse_iovnbd(s_dataset_path, v_dataset_path):
    """
    Parses the actual IO-VNBD dataset CSV files once downloaded via git-lfs.
    S-Dataset: Smartphone IMU/GPS
    V-Dataset: Vehicle Ground Truth Odometry/GPS
    """
    print(f"Loading real IO-VNBD data from {s_dataset_path}")
    
    try:
        # Load the real dataset
        df = pd.read_csv(s_dataset_path, encoding='latin1')
        df.columns = df.columns.str.strip()
        
        # In case encoding messed up the m/s² symbol, we will rename using substring
        # Since df.columns is just strings
        def rename_col(cols):
            new_cols = []
            for c in cols:
                if 'LATITUDE' in c: new_cols.append('gnss_lat')
                elif 'LONGITUDE' in c: new_cols.append('gnss_lon')
                elif 'SPEED' in c: new_cols.append('true_velocity_kmh')
                elif 'TIME' in c: new_cols.append('timestamp')
                elif 'ACCELEROMETER X' in c: new_cols.append('acc_x')
                elif 'ACCELEROMETER Y' in c: new_cols.append('acc_y')
                elif 'ACCELEROMETER Z' in c: new_cols.append('acc_z')
                elif 'GRAVITY X' in c: new_cols.append('grav_x')
                elif 'GRAVITY Y' in c: new_cols.append('grav_y')
                elif 'GRAVITY Z' in c: new_cols.append('grav_z')
                elif 'GYROSCOPE X' in c: new_cols.append('gyro_x')
                elif 'GYROSCOPE Y' in c: new_cols.append('gyro_y')
                elif 'GYROSCOPE Z' in c: new_cols.append('gyro_z')
                else: new_cols.append(c)
            return new_cols
            
        df.columns = rename_col(df.columns)
        
        # Ensure we have the base required columns for the EKF and AI model
        required = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'gnss_lat', 'gnss_lon', 'grav_x', 'grav_y', 'grav_z']
        for col in required:
            if col not in df.columns:
                print(f"Warning: Missing column {col}, filling with zeros.")
                df[col] = 0.0
                
        # Real GPS updates at 1Hz, but IMU is 10Hz. The dataset forward-fills the GPS.
        # We need to remove the duplicates and interpolate to make the UI movement smooth.
        for col in ['gnss_lat', 'gnss_lon']:
            mask = df[col] != df[col].shift(1)
            # We must never set the very first row to NaN
            mask.iloc[0] = True 
            df.loc[~mask, col] = np.nan
            df[col] = df[col].interpolate(method='linear').bfill().ffill()
            
        if 'true_velocity_kmh' in df.columns:
            mask = df['true_velocity_kmh'] != df['true_velocity_kmh'].shift(1)
            mask.iloc[0] = True
            df.loc[~mask, 'true_velocity_kmh'] = np.nan
            # The SPEED column in Android is already m/s, not km/h!
            df['true_velocity'] = df['true_velocity_kmh'].interpolate(method='linear').bfill().ffill()
        else:
            df['true_velocity'] = 15.0
            
        # Trim to a 2000-point segment so the demo plays out in ~1 minute instead of 45 minutes
        # We start a bit into the dataset to skip any initial standing still
        if len(df) > 3000:
            df = df.iloc[1000:3000].reset_index(drop=True)
            
        # Simulate a GNSS blackout for the IDR demonstration
        # We will blank out GNSS data for the middle 30% of the route
        total_len = len(df)
        start_blackout = int(total_len * 0.4)
        end_blackout = int(total_len * 0.7)
        
        df['gnss_status'] = 1.0
        df.loc[start_blackout:end_blackout, 'gnss_status'] = 0.0
        
        # Ground truth for Map
        df['true_lat'] = df['gnss_lat']
        df['true_lon'] = df['gnss_lon']
        
        # Apply blackout to measured GNSS
        df.loc[start_blackout:end_blackout, 'gnss_lat'] = np.nan
        df.loc[start_blackout:end_blackout, 'gnss_lon'] = np.nan
        
        return df
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

if __name__ == "__main__":
    base_dir = "data/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset"
    s_path = os.path.join(base_dir, "S-Dataset/S-S1.csv")
    v_path = os.path.join(base_dir, "V-Dataset/V-S1.csv")
    
    if os.path.exists(s_path) and os.path.getsize(s_path) > 1000:
        # Generate Testing Dataset
        print("Generating Testing Dataset (S-S1.csv)...")
        df1 = parse_iovnbd(s_path, v_path)
        df1.to_csv("data/real_route_processed.csv", index=False)
        
        # Generate Training Dataset to prevent overfitting
        s_path2 = os.path.join(base_dir, "S-Dataset/S-S2.csv")
        v_path2 = os.path.join(base_dir, "V-Dataset/V-S2.csv")
        if os.path.exists(s_path2):
            print("Generating Training Dataset (S-S2.csv)...")
            df2 = parse_iovnbd(s_path2, v_path2)
            df2.to_csv("data/train_route_processed.csv", index=False)
            
    else:
        print("Real dataset not fully downloaded yet via Git LFS.")
