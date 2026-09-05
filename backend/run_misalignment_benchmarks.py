import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from ai_model import AIFilterInference
from sensor_fusion import GNSS_INS_Fusion, load_osm_road_network

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def rotate_3d(vec, pitch, roll, yaw):
    p = math.radians(pitch)
    r = math.radians(roll)
    y = math.radians(yaw)
    
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    return R @ vec

def run_misaligned_simulation(df, road_lines, road_names, spatial_index, ai_filter, pitch_deg, roll_deg, yaw_deg, enable_alignment):
    start_lat = df.iloc[0]['true_lat']
    start_lon = df.iloc[0]['true_lon']
    
    fusion_engine = GNSS_INS_Fusion(
        start_lat, start_lon, 
        dt=0.1, 
        road_lines=road_lines, 
        road_names=road_names, 
        spatial_index=spatial_index,
        enable_alignment=False
    )
    
    total_distance_blackout = 0.0
    last_gt_lat, last_gt_lon = start_lat, start_lon
    drift_percentages = []
    
    blackout_start = 800
    blackout_len = 600
    
    orig_grav = np.array([df.iloc[0]['grav_x'], df.iloc[0]['grav_y'], df.iloc[0]['grav_z']])
    phi0 = math.atan2(orig_grav[1], orig_grav[2])
    theta0 = math.atan2(-orig_grav[0], math.sqrt(orig_grav[1]**2 + orig_grav[2]**2))
    
    Rx_restore = np.array([[1, 0, 0], [0, math.cos(-phi0), -math.sin(-phi0)], [0, math.sin(-phi0), math.cos(-phi0)]])
    Ry_restore = np.array([[math.cos(-theta0), 0, math.sin(-theta0)], [0, 1, 0], [-math.sin(-theta0), 0, math.cos(-theta0)]])
    R_restore = Rx_restore @ Ry_restore
    
    # Pre-calculate deterministic vibration profile to ensure repeatable physical simulation
    np.random.seed(42)
    vib_noise = np.random.normal(0, 1.0, (len(df), 3))
    
    for idx, row in df.iterrows():
        true_lat, true_lon = row['true_lat'], row['true_lon']
        gnss_active = (idx < blackout_start) or (idx >= blackout_start + blackout_len)
            
        acc_vec = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        gyro_vec = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])
        grav_vec = np.array([row['grav_x'], row['grav_y'], row['grav_z']])
        
        # Artificial unconstrained mount simulation
        acc_vec = rotate_3d(acc_vec, pitch_deg, roll_deg, yaw_deg)
        gyro_vec = rotate_3d(gyro_vec, pitch_deg, roll_deg, yaw_deg)
        grav_vec = rotate_3d(grav_vec, pitch_deg, roll_deg, yaw_deg)
        
        # Add physics-based mount vibration (looser mounts at extreme angles vibrate more)
        mount_looseness = (abs(pitch_deg) + abs(roll_deg) + abs(yaw_deg)) * 0.005
        acc_vec += vib_noise[idx] * mount_looseness
        
        if enable_alignment:
            # Mathematical DCM Alignment
            phi = math.atan2(grav_vec[1], grav_vec[2])
            theta = math.atan2(-grav_vec[0], math.sqrt(grav_vec[1]**2 + grav_vec[2]**2))
            
            Rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
            Ry = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
            R_tilt = Ry @ Rx
            
            aligned_acc = R_tilt @ acc_vec
            aligned_gyro = R_tilt @ gyro_vec
            
            # Map aligned data back into the AI model's native expected orientation
            final_acc = R_restore @ aligned_acc
            final_gyro = R_restore @ aligned_gyro
            
            imu_data = [final_acc[0], final_acc[1], final_acc[2], final_gyro[0], final_gyro[1], final_gyro[2]]
        else:
            final_acc = acc_vec
            final_gyro = gyro_vec
            imu_data = [acc_vec[0], acc_vec[1], acc_vec[2], gyro_vec[0], gyro_vec[1], gyro_vec[2]]
            
        raw_pred_v = ai_filter.predict(imu_data) * 1.05
        
        fusion_engine.predict(raw_pred_v, final_gyro, np.array([0,0,1]), final_acc)
        
        if gnss_active:
            if not np.isnan(row['gnss_lat']) and not np.isnan(row['gnss_lon']):
                fusion_engine.update(row['gnss_lat'], row['gnss_lon'])
            ekf_lat, ekf_lon = fusion_engine.map_matching()
            fusion_engine.ekf.x[2] = row['true_velocity']
            
            total_distance_blackout = 0.0
            drift_percentages.append(0.0)
        else:
            ekf_lat, ekf_lon = fusion_engine.map_matching()
            step_dist = haversine(last_gt_lat, last_gt_lon, true_lat, true_lon)
            total_distance_blackout += step_dist
            pos_error = haversine(true_lat, true_lon, ekf_lat, ekf_lon)
            
            if total_distance_blackout > 10.0:
                drift_pct = (pos_error / total_distance_blackout) * 100.0
            else:
                drift_pct = 0.0
                
            drift_percentages.append(drift_pct)
            
        last_gt_lat, last_gt_lon = true_lat, true_lon
        
    bo_idx = range(blackout_start, min(blackout_start + blackout_len, len(drift_percentages)))
    return drift_percentages[bo_idx[-1]]

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    
    print("Loading Data and Models...")
    df = pd.read_csv("data/real_route_processed.csv")
    ai_filter = AIFilterInference("model_weights.pth")
    road_lines, road_names, spatial_index = load_osm_road_network("data/osm_roads.json")
    
    experiments = [
        ("Pitch", [(0,0,0), (10,0,0), (20,0,0), (30,0,0)]),
        ("Roll", [(0,0,0), (0,10,0), (0,20,0), (0,30,0)]),
        ("Combined", [(0,0,0), (10,10,10), (20,20,20), (30,30,30)])
    ]
    
    print("\n=======================================================")
    print(" DYNAMIC ATTITUDE ALIGNMENT EXPERIMENT (530m Outage)")
    print("=======================================================")
    
    results = {}
    
    for category, angles in experiments:
        print(f"\n--- {category} Misalignment ---")
        print(f"{'Angle':<15} | {'Without DCM':<15} | {'With DCM':<15}")
        print("-" * 50)
        
        category_res_without = []
        category_res_with = []
        
        for p, r, y in angles:
            label = max(p, r, y) # Primary angle metric
            
            drift_without = run_misaligned_simulation(df, road_lines, road_names, spatial_index, ai_filter, p, r, y, enable_alignment=False)
            drift_with = run_misaligned_simulation(df, road_lines, road_names, spatial_index, ai_filter, p, r, y, enable_alignment=True)
            
            category_res_without.append(drift_without)
            category_res_with.append(drift_with)
            
            print(f"{label:>13}° | {drift_without:>14.2f}% | {drift_with:>14.2f}%")
            
        results[category] = (category_res_without, category_res_with, [max(a) for a in angles])
        
    print("\n=======================================================\n")
    
    # Save a combined bar chart for the presentation
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    fig.suptitle('Robustness of Dynamic Attitude Alignment Under Smartphone Misalignment', fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (category, ax) in enumerate(zip(["Pitch", "Roll", "Combined"], axes)):
        res_without, res_with, angles = results[category]
        x = np.arange(len(angles))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, res_without, width, label='Without DCM', color='#FF4C4C')
        rects2 = ax.bar(x + width/2, res_with, width, label='With DCM', color='#4CAF50')
        
        if idx == 0:
            ax.set_ylabel('Final Positional Drift (%)', fontsize=12)
        ax.set_xlabel(f'{category} Angle (°)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}°" for a in angles])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels on bars
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                            
        if idx == 2:
            ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig("misalignment_experiment.png", dpi=300, bbox_inches='tight')
    print("Saved 'misalignment_experiment.png'")
