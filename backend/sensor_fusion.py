import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import math
import json
from shapely.geometry import Point, LineString
import rtree

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def dist_meters(lat1, lon1, lat2, lon2):
    return haversine(lat1, lon1, lat2, lon2)

def load_osm_road_network(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return [], [], None
        
    road_lines = []
    road_names = []
    idx = rtree.index.Index()
    
    for i, element in enumerate(data.get('elements', [])):
        if element['type'] == 'way' and 'geometry' in element:
            coords = [(pt['lon'], pt['lat']) for pt in element['geometry']]
            if len(coords) >= 2:
                line = LineString(coords)
                road_lines.append(line)
                name = element.get('tags', {}).get('name', f'Road_{element.get("id", i)}')
                road_names.append(name)
                idx.insert(i, line.bounds)
                
    return road_lines, road_names, idx

class GNSS_INS_Fusion:
    def __init__(self, start_lat, start_lon, dt=0.1, road_lines=None, road_names=None, spatial_index=None, enable_alignment=True):
        self.dt = dt
        self.start_lat = start_lat
        self.enable_alignment = enable_alignment
        
        self.ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2)
        
        self.ekf.x = np.array([start_lat, start_lon, 0.0, 0.0, 0.0])
        self.ekf.P = np.eye(5) * 1.0
        self.ekf.Q = np.eye(5) * 0.1
        self.ekf.R = np.eye(2) * 5.0
        
        self.m_per_deg_lat = 111320.0
        self.prev_gnss_lat = start_lat
        self.prev_gnss_lon = start_lon
        
        self.acc_mag_buffer = []
        self.gnss_rejection_count = 0
        
        self.yaw_misalignment = 0.0
        self.DCM = np.eye(3)
        
        self.hmm_state = []
        self.last_ekf_pos = None
        
        self.road_lines = road_lines or []
        self.road_names = road_names or []
        self.spatial_index = spatial_index

    def align_sensors(self, acc_vec, gyro_vec, grav_vec, is_moving_fast=False):
        if not self.enable_alignment:
            return acc_vec, gyro_vec
            
        grav_norm = np.linalg.norm(grav_vec)
        if grav_norm < 1.0:
            return acc_vec, gyro_vec 
            
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
        acc_level = R_tilt @ acc_vec
        
        acc_horiz_mag = np.sqrt(acc_level[0]**2 + acc_level[1]**2)
        if is_moving_fast and acc_horiz_mag > 0.5:
            measured_yaw = np.arctan2(acc_level[1], acc_level[0])
            diff = measured_yaw - self.yaw_misalignment
            while diff > np.pi: diff -= 2 * np.pi
            while diff < -np.pi: diff += 2 * np.pi
            self.yaw_misalignment += 0.02 * diff
            
        psi = -self.yaw_misalignment
        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        self.DCM = Rz @ R_tilt
        aligned_acc = self.DCM @ acc_vec
        aligned_gyro = self.DCM @ gyro_vec
        
        return aligned_acc, aligned_gyro

    def predict(self, ai_velocity, gyro_vec, grav_vec, acc_vec):
        lat, lon, v_fwd, v_lat, theta = self.ekf.x
        
        acc_mag = np.linalg.norm(acc_vec)
        self.acc_mag_buffer.append(acc_mag)
        if len(self.acc_mag_buffer) > 10:
            self.acc_mag_buffer.pop(0)
            
        is_stationary = False
        if len(self.acc_mag_buffer) == 10:
            acc_var = np.var(self.acc_mag_buffer)
            if acc_var < 0.01 and abs(gyro_vec[2]) < 0.02 and ai_velocity < 1.0:
                is_stationary = True
        
        if not self.enable_alignment:
            true_yaw_rate = gyro_vec[2]
        else:
            grav_norm = np.linalg.norm(grav_vec)
            if grav_norm < 1.0:
                true_yaw_rate = gyro_vec[2]
            else:
                true_yaw_rate = np.dot(gyro_vec, grav_vec) / grav_norm
            
        if is_stationary:
            v_fwd_pred = 0.0
            v_lat_pred = 0.0
            theta_pred = theta
            self.ekf.x[2] = 0.0
            self.ekf.x[3] = 0.0
        else:
            v_fwd_pred = v_fwd 
            v_lat_pred = v_lat
            theta_pred = theta - (true_yaw_rate * self.dt)
            
        m_per_deg_lon = 111320.0 * math.cos(math.radians(self.start_lat))
        
        v_N = v_fwd_pred * math.cos(theta_pred) - v_lat_pred * math.sin(theta_pred)
        v_E = v_fwd_pred * math.sin(theta_pred) + v_lat_pred * math.cos(theta_pred)
        
        new_lat = lat + (v_N * self.dt) / self.m_per_deg_lat
        new_lon = lon + (v_E * self.dt) / m_per_deg_lon
        
        self.ekf.x = np.array([new_lat, new_lon, v_fwd_pred, v_lat_pred, theta_pred])
        
        self.ekf.F = np.eye(5)
        self.ekf.F[0, 2] = (math.cos(theta_pred) * self.dt) / self.m_per_deg_lat
        self.ekf.F[0, 3] = -(math.sin(theta_pred) * self.dt) / self.m_per_deg_lat
        self.ekf.F[0, 4] = (-v_fwd_pred * math.sin(theta_pred) - v_lat_pred * math.cos(theta_pred)) * self.dt / self.m_per_deg_lat
        
        self.ekf.F[1, 2] = (math.sin(theta_pred) * self.dt) / m_per_deg_lon
        self.ekf.F[1, 3] = (math.cos(theta_pred) * self.dt) / m_per_deg_lon
        self.ekf.F[1, 4] = (v_fwd_pred * math.cos(theta_pred) - v_lat_pred * math.sin(theta_pred)) * self.dt / m_per_deg_lon
        
        # Rigorous non-linear P update manually to prevent filterpy overwriting x
        self.ekf.P = self.ekf.F @ self.ekf.P @ self.ekf.F.T + self.ekf.Q
        
        # 1 & 2. AI Velocity & NHC Measurement Updates
        if not is_stationary:
            z_kinematic = np.array([ai_velocity, 0.0])
            def H_kinematic(x):
                H = np.zeros((2, 5))
                H[0, 2] = 1.0  
                H[1, 3] = 1.0  
                return H
            def hx_kinematic(x):
                return np.array([x[2], x[3]])
            R_kinematic = np.array([[0.5, 0.0], [0.0, 0.01]])
            self.ekf.update(z_kinematic, HJacobian=H_kinematic, Hx=hx_kinematic, R=R_kinematic)
        else:
            z_kinematic = np.array([0.0, 0.0])
            def H_kinematic(x):
                H = np.zeros((2, 5))
                H[0, 2] = 1.0
                H[1, 3] = 1.0
                return H
            def hx_kinematic(x):
                return np.array([x[2], x[3]])
            R_kinematic = np.array([[0.01, 0.0], [0.0, 0.01]])
            self.ekf.update(z_kinematic, HJacobian=H_kinematic, Hx=hx_kinematic, R=R_kinematic)
        
        return self.ekf.x[0], self.ekf.x[1]

    def map_matching(self):
        if not self.spatial_index:
            return self.ekf.x[0], self.ekf.x[1]
            
        lat, lon = self.ekf.x[0], self.ekf.x[1]
        
        if self.last_ekf_pos:
            dr_dist = dist_meters(self.last_ekf_pos[0], self.last_ekf_pos[1], lat, lon)
            if dr_dist < 0.2:
                return lat, lon
                
        p = Point(lon, lat)
        search_radius = 0.001
        bbox = (lon - search_radius, lat - search_radius, lon + search_radius, lat + search_radius)
        
        candidate_indices = list(self.spatial_index.intersection(bbox))
        if not candidate_indices:
            return lat, lon
            
        sigma = 10.0
        beta = 5.0
        
        new_hmm_state = []
        best_candidate = None
        max_log_prob = -float('inf')
        
        for idx in candidate_indices:
            line = self.road_lines[idx]
            proj_dist_deg = line.project(p)
            proj_p = line.interpolate(proj_dist_deg)
            snap_lon, snap_lat = proj_p.x, proj_p.y
            
            dist_to_road = dist_meters(lat, lon, snap_lat, snap_lon)
            emission_log_prob = -0.5 * (dist_to_road / sigma)**2
            
            best_trans_log_prob = -float('inf')
            
            if not self.hmm_state:
                total_log_prob = emission_log_prob
            else:
                for prev_state in self.hmm_state:
                    prev_lat, prev_lon = prev_state['lat'], prev_state['lon']
                    map_dist = dist_meters(prev_lat, prev_lon, snap_lat, snap_lon)
                    
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
                
        for state in new_hmm_state:
            state['log_prob'] -= max_log_prob
            
        self.hmm_state = new_hmm_state
        self.last_ekf_pos = (lat, lon)
        
        if best_candidate:
            snap_lat, snap_lon, nearest_road, proj_dist = best_candidate
            
            dist_to_best = dist_meters(lat, lon, snap_lat, snap_lon)
            if dist_to_best < 30.0:
                road_length = nearest_road.length
                look_ahead_dist = min(proj_dist + 0.0001, road_length)
                look_behind_dist = max(proj_dist - 0.0001, 0.0)
                look_ahead_pt = nearest_road.interpolate(look_ahead_dist)
                look_behind_pt = nearest_road.interpolate(look_behind_dist)
                
                m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
                dy_fwd = (look_ahead_pt.y - look_behind_pt.y) * 111320.0
                dx_fwd = (look_ahead_pt.x - look_behind_pt.x) * m_per_deg_lon
                
                if math.sqrt(dx_fwd**2 + dy_fwd**2) > 0.05:
                    road_heading = math.atan2(dx_fwd, dy_fwd)
                    
                    diff_fwd = road_heading - self.ekf.x[4]
                    while diff_fwd > math.pi: diff_fwd -= 2 * math.pi
                    while diff_fwd < -math.pi: diff_fwd += 2 * math.pi
                    
                    rev_heading = road_heading + math.pi
                    diff_rev = rev_heading - self.ekf.x[4]
                    while diff_rev > math.pi: diff_rev -= 2 * math.pi
                    while diff_rev < -math.pi: diff_rev += 2 * math.pi
                    
                    target_heading = road_heading if abs(diff_fwd) <= abs(diff_rev) else rev_heading
                    
                    # Rigorous EKF heading constraint
                    def H_hmm_heading(x):
                        return np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
                    def hx_hmm_heading(x):
                        return np.array([x[4]])
                        
                    self.ekf.dim_z = 1
                    diff = target_heading - self.ekf.x[4]
                    while diff > math.pi: diff -= 2 * math.pi
                    while diff < -math.pi: diff += 2 * math.pi
                    self.ekf.update(np.array([self.ekf.x[4] + diff]), HJacobian=H_hmm_heading, Hx=hx_hmm_heading, R=np.array([[0.05]]))
                    self.ekf.dim_z = 2
                    
                    while self.ekf.x[4] > math.pi: self.ekf.x[4] -= 2 * math.pi
                    while self.ekf.x[4] < -math.pi: self.ekf.x[4] += 2 * math.pi
                    
            return snap_lat, snap_lon
            
        return lat, lon

    def update(self, gnss_lat, gnss_lon):
        z = np.array([gnss_lat, gnss_lon])
        
        def H_jacobian(x):
            H = np.zeros((2, 5))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            return H
            
        def hx(x):
            return np.array([x[0], x[1]])
            
        H_val = H_jacobian(self.ekf.x)
        y = z - hx(self.ekf.x)
        S = H_val @ self.ekf.P @ H_val.T + self.ekf.R
        
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_sq = y.T @ S_inv @ y
            
            if mahalanobis_sq > 9.21:
                self.gnss_rejection_count += 1
                if self.gnss_rejection_count > 5:
                    inflated_R = self.ekf.R * (mahalanobis_sq / 9.21)
                    self.ekf.update(z, HJacobian=H_jacobian, Hx=hx, R=inflated_R)
                    self.gnss_rejection_count = 0
                else:
                    return self.ekf.x[0], self.ekf.x[1]
            else:
                self.gnss_rejection_count = 0
                self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)
        except np.linalg.LinAlgError:
            self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)
        
        delta_lat = gnss_lat - self.prev_gnss_lat
        delta_lon = gnss_lon - self.prev_gnss_lon
        m_per_deg_lon = 111320.0 * math.cos(math.radians(gnss_lat))
        dy = delta_lat * self.m_per_deg_lat
        dx = delta_lon * m_per_deg_lon
        
        if math.sqrt(dx**2 + dy**2) > 0.05:
            measured_heading = math.atan2(dx, dy)
            if self.ekf.x[4] == 0.0:
                self.ekf.x[4] = measured_heading
            else:
                diff = measured_heading - self.ekf.x[4]
                while diff > math.pi: diff -= 2 * math.pi
                while diff < -math.pi: diff += 2 * math.pi
                
                def H_hdg(x):
                    return np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
                def hx_hdg(x):
                    return np.array([x[4], 0.0])
                
                self.ekf.update(np.array([self.ekf.x[4] + diff, 0.0]), HJacobian=H_hdg, Hx=hx_hdg, R=np.array([[0.1, 0.0], [0.0, 99999.0]]))
                
                while self.ekf.x[4] > math.pi: self.ekf.x[4] -= 2 * math.pi
                while self.ekf.x[4] < -math.pi: self.ekf.x[4] += 2 * math.pi
            
        self.prev_gnss_lat = gnss_lat
        self.prev_gnss_lon = gnss_lon
        
        return self.ekf.x[0], self.ekf.x[1]
