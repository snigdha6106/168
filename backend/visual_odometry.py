import cv2
import numpy as np

class VisualOdometry:
    """
    Monocular Visual Odometry (VIO) Engine.
    Uses Lucas-Kanade Optical Flow to estimate forward vehicle velocity
    from a live smartphone camera feed.
    """
    def __init__(self, focal_length=700, camera_height=1.2):
        self.focal_length = focal_length       # Assumed focal length in pixels
        self.camera_height = camera_height     # Height of smartphone camera from road (meters)
        
        self.prev_gray = None
        self.prev_points = None
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Shi-Tomasi corner detection parameters
        self.feature_params = dict(
            maxCorners=300,
            qualityLevel=0.05,
            minDistance=10,
            blockSize=7
        )

    def process_frame(self, frame_bgr, dt):
        """
        Process an incoming camera frame and return estimated forward velocity.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Crop to the bottom half of the image (road surface)
        h, w = gray.shape
        roi_gray = gray[int(h/2):, :]
        
        velocity_estimate = None

        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) < 10:
            # Detect new features to track
            self.prev_points = cv2.goodFeaturesToTrack(roi_gray, mask=None, **self.feature_params)
            self.prev_gray = roi_gray
            return None # Not enough history to calculate speed yet

        # Calculate Optical Flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, roi_gray, self.prev_points, None, **self.lk_params
        )

        # Select good points
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]

        if len(good_new) > 10:
            # Calculate pixel displacement (flow vectors)
            flow_vectors = good_new - good_old
            
            # Extract dominant downward vertical flow (as road moves "down" in the camera view when driving forward)
            # In image coordinates, y increases downwards.
            y_flows = flow_vectors[:, 1]
            
            # Filter out noise (only consider positive flow i.e., moving forward)
            forward_flows = y_flows[y_flows > 0]
            
            if len(forward_flows) > 0:
                median_flow_px = np.median(forward_flows)
                
                # Monocular Scale Approximation:
                # Z = (f * Y) / y  => dZ/dt = (f * Y) / y^2 * (dy/dt)
                # Roughly estimates speed assuming features are on the flat road.
                # v = (displacement_px / dt) * (camera_height / focal_length)
                
                velocity_estimate = (median_flow_px / dt) * (self.camera_height / self.focal_length)
                
                # Apply physically realistic bounds (0 to 40 m/s)
                velocity_estimate = np.clip(velocity_estimate, 0.0, 40.0)

            self.prev_points = good_new.reshape(-1, 1, 2)
            self.prev_gray = roi_gray
        else:
            # Lost tracking, reset
            self.prev_points = None
            self.prev_gray = roi_gray
            
        return velocity_estimate
