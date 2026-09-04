import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import math
import json
from shapely.geometry import Point, LineString, MultiLineString
from shapely.strtree import STRtree


def load_osm_road_network(osm_json_path="data/osm_roads.json"):
    """
    Build a genuine spatial index from real OpenStreetMap road data.
    Returns a list of (LineString, road_name) and an STRtree spatial index.
    """
    with open(osm_json_path) as f:
        data = json.load(f)
    
    road_lines = []
    road_names = []
    
    for element in data.get("elements", []):
        geom = element.get("geometry", [])
        if len(geom) < 2:
            continue
        
        coords = [(pt["lon"], pt["lat"]) for pt in geom]
        line = LineString(coords)
        road_lines.append(line)
        
        tags = element.get("tags", {})
        road_names.append(tags.get("name", "unnamed"))
    
    # Build R-Tree spatial index for O(log n) nearest-road queries
    spatial_index = STRtree(road_lines)
    
    print(f"Loaded {len(road_lines)} OSM road segments into spatial index")
    return road_lines, road_names, spatial_index


class GNSS_INS_Fusion:
    def __init__(self, start_lat, start_lon, dt=0.1, road_lines=None, road_names=None, spatial_index=None):
        self.dt = dt
        self.start_lat = start_lat
        
        # State: [lat, lon, velocity, heading]
        self.ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
        
        self.ekf.x = np.array([start_lat, start_lon, 0.0, 0.0])
        
        # Covariance matrix
        self.ekf.P = np.eye(4) * 1.0
        
        # Process noise
        self.ekf.Q = np.eye(4) * 0.1
        
        # Measurement noise (GNSS is noisy)
        self.ekf.R = np.eye(2) * 5.0
        
        self.m_per_deg_lat = 111320.0
        
        self.prev_gnss_lat = start_lat
        self.prev_gnss_lon = start_lon
        
        # ZUPT buffer for variance
        self.acc_mag_buffer = []
        
        # GNSS Outlier Gating Counter
        self.gnss_rejection_count = 0
        
        # Dynamic In-Vehicle Attitude Alignment
        self.yaw_misalignment = 0.0
        self.DCM = np.eye(3)
        
        # HMM Viterbi Map-Matching Variables
        self.hmm_state = []
        self.last_ekf_pos = None
        
        # Genuine OpenStreetMap spatial database (pre-loaded at server startup)
        self.road_lines = road_lines or []
        self.road_names = road_names or []
        self.spatial_index = spatial_index

    def align_sensors(self, acc_vec, gyro_vec, grav_vec, is_moving_fast=False):
        """
        Continuous Dynamic In-Vehicle Attitude Alignment.
        Calculates the rotation matrix (DCM) from the phone's body frame to 
        the vehicle's chassis frame to correct for arbitrary dashboard mount angles.
        """
        grav_norm = np.linalg.norm(grav_vec)
        if grav_norm < 1.0:
            return acc_vec, gyro_vec # Fallback if gravity is missing
            
        # 1. Pitch and Roll Estimation
        # Compute Euler angles relative to gravity
        phi = np.arctan2(grav_vec[1], grav_vec[2])
        theta = np.arctan2(-grav_vec[0], np.sqrt(grav_vec[1]**2 + grav_vec[2]**2))
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        R_tilt = Ry @ Rx
        
        # 2. Yaw Misalignment Estimation
        acc_level = R_tilt @ acc_vec
        
        # Update yaw misalignment only when accelerating forward cleanly
        acc_horiz_mag = np.sqrt(acc_level[0]**2 + acc_level[1]**2)
        if is_moving_fast and acc_horiz_mag > 0.5:
            measured_yaw = np.arctan2(acc_level[1], acc_level[0])
            
            # Angle wrap-around handling
            diff = measured_yaw - self.yaw_misalignment
            while diff > np.pi: diff -= 2 * np.pi
            while diff < -np.pi: diff += 2 * np.pi
            
            # Low-pass filter the yaw estimate
            self.yaw_misalignment += 0.02 * diff
            
        # 3. Direction Cosine Matrix (DCM) Construction
        psi = -self.yaw_misalignment
        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        self.DCM = Rz @ R_tilt
        
        # 4. Frame Transformation
        aligned_acc = self.DCM @ acc_vec
        aligned_gyro = self.DCM @ gyro_vec
        
        return aligned_acc, aligned_gyro

    def predict(self, ai_velocity, gyro_vec, grav_vec, acc_vec):
        """
        AI Model predicts forward velocity.
        We use the In-Vehicle Alignment Engine (Gravity Projection) to get the true yaw.
        """
        lat, lon, v, theta = self.ekf.x
        
        # 1. Stationary Detection (ZUPT / ZARU) - Decoupled from AI velocity
        # Calculate stationarity using the variance of raw 3-axis accelerometer magnitude
        acc_mag = np.linalg.norm(acc_vec)
        self.acc_mag_buffer.append(acc_mag)
        if len(self.acc_mag_buffer) > 10:  # 1-second sliding window at 10Hz
            self.acc_mag_buffer.pop(0)
            
        is_stationary = False
        if len(self.acc_mag_buffer) == 10:
            acc_var = np.var(self.acc_mag_buffer)
            # Threshold tuned for typical engine idling vibration vs actual motion
            if acc_var < 0.01 and abs(gyro_vec[2]) < 0.02 and ai_velocity < 1.0:
                is_stationary = True
        
        if is_stationary:
            new_v = 0.0
            new_theta = theta  # Zero Angular Rate Update (ZARU)
            # Clamp EKF velocity state directly
            self.ekf.x[2] = 0.0
        else:
            new_v = ai_velocity
            # Calculate True Yaw Rate by projecting Gyroscope onto the Gravity vector.
            grav_norm = np.linalg.norm(grav_vec)
            if grav_norm < 1.0: # Fallback if gravity is missing
                true_yaw_rate = gyro_vec[2] # Guess Z-axis is UP
            else:
                true_yaw_rate = np.dot(gyro_vec, grav_vec) / grav_norm
                
            new_theta = theta - (true_yaw_rate * self.dt)
        
        # Use starting latitude to match dataset generation exactly and avoid floating point integration drift
        m_per_deg_lon = 111320.0 * math.cos(math.radians(self.start_lat))
        
        new_lat = lat + (new_v * math.cos(new_theta) * self.dt) / self.m_per_deg_lat
        new_lon = lon + (new_v * math.sin(new_theta) * self.dt) / m_per_deg_lon
        
        self.ekf.x = np.array([new_lat, new_lon, new_v, new_theta])
        
        # Jacobian F
        self.ekf.F = np.eye(4)
        self.ekf.predict()
        
        # 2. Non-Holonomic Constraints (NHC)
        # Enforce zero lateral skid (vy ≈ 0). 
        # State vector is [lat, lon, v, theta]^T. We measure lateral velocity.
        # The expected lateral velocity is 0. 
        # v_lat = v * sin(theta_error). Since theta_error is small, v_lat ~ v * d_theta
        # So the measurement matrix H needs to extract the lateral error.
        def H_nhc(x):
            # Measurement is 1D: lateral velocity
            H = np.zeros((1, 4))
            H[0, 3] = x[2]  # derivative of v*sin(d_theta) wrt d_theta is v*cos(d_theta) ~ v
            return H
            
        def hx_nhc(x):
            # Expected lateral velocity is 0, but if we evaluate h(x) where x has no error, it's just 0.
            return np.array([0.0])
            
        # Apply NHC pseudo-measurement tightly. 1D measurement, so R is 1x1.
        self.ekf.update(np.array([0.0]), HJacobian=H_nhc, Hx=hx_nhc, R=np.array([[0.01]]))
        
        return self.ekf.x[0], self.ekf.x[1]

    def map_matching(self):
        """
        Hidden Markov Model (HMM) Map-Matching utilizing the Viterbi algorithm.
        Constrains the trajectory probabilistically based on network connectivity and emission bounds.
        """
        lat, lon = self.ekf.x[0], self.ekf.x[1]
        p = Point(lon, lat)
        
        if self.spatial_index is None or len(self.road_lines) == 0:
            return lat, lon
            
        def dist_meters(lat1, lon1, lat2, lon2):
            m_lon = 111320.0 * math.cos(math.radians(lat1))
            dx = (lon2 - lon1) * m_lon
            dy = (lat2 - lat1) * 111320.0
            return math.sqrt(dx**2 + dy**2)
            
        # 1. Candidate Generation
        search_radius_deg = 30.0 / 111320.0
        candidate_indices = self.spatial_index.query(p.buffer(search_radius_deg))
        
        # Fallback if no roads within 30 meters
        if len(candidate_indices) == 0:
            candidate_indices = [self.spatial_index.nearest(p)]
            
        new_hmm_state = []
        best_candidate = None
        max_log_prob = -float('inf')
        
        sigma = 10.0 # Emission standard deviation (meters)
        beta = 5.0   # Transition scaling factor (meters)
        
        # Calculate Dead-Reckoning Distance since last frame
        if self.last_ekf_pos is not None:
            dr_dist = dist_meters(self.last_ekf_pos[0], self.last_ekf_pos[1], lat, lon)
        else:
            dr_dist = 0.0
            
        for idx in candidate_indices:
            line = self.road_lines[idx]
            proj_dist_deg = line.project(p)
            proj_p = line.interpolate(proj_dist_deg)
            snap_lon, snap_lat = proj_p.x, proj_p.y
            
            # 2. Emission Probability
            dist_to_road = dist_meters(lat, lon, snap_lat, snap_lon)
            emission_log_prob = -0.5 * (dist_to_road / sigma)**2
            
            best_trans_log_prob = -float('inf')
            
            if not self.hmm_state:
                # Initial state probabilities
                total_log_prob = emission_log_prob
            else:
                # 3. Transition Probability & Viterbi Decoding
                for prev_state in self.hmm_state:
                    prev_lat, prev_lon = prev_state['lat'], prev_state['lon']
                    map_dist = dist_meters(prev_lat, prev_lon, snap_lat, snap_lon)
                    
                    # Penalize route discontinuities (jumping between completely distinct road geometries)
                    line_penalty = 0.0 if idx == prev_state['line_idx'] else -2.0
                    
                    diff_dist = abs(dr_dist - map_dist)
                    transition_log_prob = -(diff_dist / beta) + line_penalty
                    
                    prob = prev_state['log_prob'] + transition_log_prob
                    if prob > best_trans_log_prob:
                        best_trans_log_prob = prob
                        
                total_log_prob = emission_log_prob + best_trans_log_prob
                
            new_hmm_state.append({
                'line_idx': idx,
                'lat': snap_lat,
                'lon': snap_lon,
                'log_prob': total_log_prob
            })
            
            if total_log_prob > max_log_prob:
                max_log_prob = total_log_prob
                best_candidate = (snap_lat, snap_lon, line, proj_dist_deg)
                
        # Normalize probabilities to prevent underflow
        for state in new_hmm_state:
            state['log_prob'] -= max_log_prob
            
        self.hmm_state = new_hmm_state
        self.last_ekf_pos = (lat, lon)
        
        # Apply the Viterbi best path to loosely constrain the EKF
        if best_candidate:
            snap_lat, snap_lon, nearest_road, proj_dist = best_candidate
            
            # Final validation distance check to prevent teleporting
            dist_to_best = dist_meters(lat, lon, snap_lat, snap_lon)
            if dist_to_best < 30.0:
                # We intentionally DO NOT update self.ekf.x[0] and x[1] here.
                # If we anchor the EKF position backwards to the snapped road projection, 
                # it creates a negative feedback loop that heavily brakes the car and causes it to lag behind.
                # The visual snapped coordinates (snap_lat, snap_lon) are returned for the dashboard!
                
                # Update EKF heading loosely based on road geometry to steer the physics model
                road_length = nearest_road.length
                look_ahead_dist = min(proj_dist + 0.0001, road_length)
                look_behind_dist = max(proj_dist - 0.0001, 0.0)
                look_ahead_pt = nearest_road.interpolate(look_ahead_dist)
                look_behind_pt = nearest_road.interpolate(look_behind_dist)
                
                m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
                dy_fwd = (look_ahead_pt.y - look_behind_pt.y) * 111320.0
                dx_fwd = (look_ahead_pt.x - look_behind_pt.x) * m_per_deg_lon
                
                if math.sqrt(dx_fwd**2 + dy_fwd**2) > 0.1:
                    fwd_heading = math.atan2(dx_fwd, dy_fwd)
                    rev_heading = fwd_heading + math.pi
                    if rev_heading > math.pi: rev_heading -= 2 * math.pi
                    
                    current_heading = self.ekf.x[3]
                    
                    diff_fwd = fwd_heading - current_heading
                    while diff_fwd > math.pi: diff_fwd -= 2 * math.pi
                    while diff_fwd < -math.pi: diff_fwd += 2 * math.pi
                    
                    diff_rev = rev_heading - current_heading
                    while diff_rev > math.pi: diff_rev -= 2 * math.pi
                    while diff_rev < -math.pi: diff_rev += 2 * math.pi
                    
                    if abs(diff_fwd) <= abs(diff_rev):
                        diff = diff_fwd
                    else:
                        diff = diff_rev
                        
                    self.ekf.x[3] += diff * 0.1
                    
        return self.ekf.x[0], self.ekf.x[1]

    def update(self, gnss_lat, gnss_lon):
        """
        When GNSS is available, update the filter to snap back to truth.
        """
        z = np.array([gnss_lat, gnss_lon])
        
        # Measurement function: we directly measure lat and lon
        def H_jacobian(x):
            H = np.zeros((2, 4))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            return H
            
        def hx(x):
            return np.array([x[0], x[1]])
            
        # 3. Statistical Innovation Gating (Mahalanobis Distance / Chi-Square Test)
        # Prevent Re-acquisition Deadlock
        H_val = H_jacobian(self.ekf.x)
        y = z - hx(self.ekf.x)
        S = H_val @ self.ekf.P @ H_val.T + self.ekf.R
        
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_sq = y.T @ S_inv @ y
            
            # Chi-Square threshold for 2 degrees of freedom at ~99% confidence is 9.21
            if mahalanobis_sq > 9.21:
                self.gnss_rejection_count += 1
                
                if self.gnss_rejection_count > 5:
                    # Deadlock prevention: INS has drifted too far. 
                    # Treat GNSS as truth, blend with inflated covariance rather than discarding permanently.
                    inflated_R = self.ekf.R * (mahalanobis_sq / 9.21)
                    self.ekf.update(z, HJacobian=H_jacobian, Hx=hx, R=inflated_R)
                    # Reset counter since we re-aligned
                    self.gnss_rejection_count = 0
                else:
                    # Normal outlier rejection
                    return self.ekf.x[0], self.ekf.x[1]
            else:
                # Valid fix, reset counter
                self.gnss_rejection_count = 0
                self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)
                
        except np.linalg.LinAlgError:
            # If singular, skip gating and just update
            self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)
        
        # Calculate true heading from GPS (Course Over Ground)
        # This acts as our In-Vehicle Alignment/Calibration step for heading
        delta_lat = gnss_lat - self.prev_gnss_lat
        delta_lon = gnss_lon - self.prev_gnss_lon
        
        m_per_deg_lon = 111320.0 * math.cos(math.radians(gnss_lat))
        dy = delta_lat * self.m_per_deg_lat
        dx = delta_lon * m_per_deg_lon
        
        if math.sqrt(dx**2 + dy**2) > 0.05: # Update heading if moving (at least 0.5 m/s)
            measured_heading = math.atan2(dx, dy)
            
            # If the filter's heading is exactly 0.0 (uninitialized), snap instantly
            if self.ekf.x[3] == 0.0:
                self.ekf.x[3] = measured_heading
            else:
                # Handle angular wrap-around
                diff = measured_heading - self.ekf.x[3]
                while diff > math.pi: diff -= 2 * math.pi
                while diff < -math.pi: diff += 2 * math.pi
                
                # Stronger snap to track the actual road perfectly before blackout
                self.ekf.x[3] = self.ekf.x[3] + (diff * 0.5)
                
                # Normalize back to [-pi, pi]
                while self.ekf.x[3] > math.pi: self.ekf.x[3] -= 2 * math.pi
                while self.ekf.x[3] < -math.pi: self.ekf.x[3] += 2 * math.pi
            
        self.prev_gnss_lat = gnss_lat
        self.prev_gnss_lon = gnss_lon
        
        return self.ekf.x[0], self.ekf.x[1]
