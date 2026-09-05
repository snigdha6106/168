import cv2
import numpy as np
import time
from visual_odometry import VisualOdometry

def run_vio_demo(video_path="dashcam.mp4", output_path="vio_output.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        print("Please place a short driving video named 'dashcam.mp4' in the backend folder!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    dt = 1.0 / fps

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize our VIO engine
    vio = VisualOdometry(focal_length=700, camera_height=1.2)
    
    print(f"Processing {video_path} at {fps} FPS...")
    
    # Create random colors for drawing optical flow tracks
    color = np.random.randint(0, 255, (300, 3))
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run the VIO engine
        velocity = vio.process_frame(frame, dt)
        
        # --- Visualization (Drawing the tracking points) ---
        if vio.prev_points is not None:
            for i, pt in enumerate(vio.prev_points):
                a, b = int(pt[0][0]), int(pt[0][1] + height/2) # Adjust for bottom-half ROI
                frame = cv2.circle(frame, (a, b), 5, color[i%300].tolist(), -1)
                
        # Draw Speed Overlay
        if velocity is not None:
            speed_kmh = velocity * 3.6
            cv2.putText(frame, f"VIO Speed: {speed_kmh:.1f} km/h", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, "Feature Tracking Active", (50, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Initializing Tracking...", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done! Demo saved to {output_path}")

if __name__ == "__main__":
    # You can change "dashcam.mp4" to 0 to use your laptop webcam live!
    run_vio_demo("dashcam.mp4", "vio_output.mp4")
