# Visual Odometry & Optical Flow Integration

While the primary GNSS-denied navigation engine relies on AI vibration profiling (IMU + LSTM), certain edge cases require hardware redundancy. For example, a perfectly smooth Electric Vehicle (EV) driving on freshly paved asphalt may not generate a strong enough vibration signature for the IMU to accurately predict speed.

To solve this, we architected a **Monocular Visual Odometry (VIO)** fallback system. This document explains the computer vision algorithms used to calculate vehicle speed purely from the smartphone's live camera feed.

---

## 1. The Core Concept (Pixel Displacement)

When looking out the side window of a moving car, objects close to the car (like pebbles on the road) blur past your vision quickly, while objects far away (like mountains) move slowly. 

The Visual Odometry engine replicates this physics principle. As the vehicle moves forward, the texture of the road surface moves *downward* and *outward* across the camera lens. By mathematically measuring exactly how many pixels the road moves in a fraction of a second, we can calculate the vehicle's physical speed in km/h.

---

## 2. Algorithmic Step-by-Step

### Step 1: Feature Extraction (Shi-Tomasi Corner Detection)
A camera cannot track speed if it is staring at a perfectly smooth, solid gray image. It needs high-contrast anchor points. 

The algorithm first isolates the bottom half of the camera frame (the Region of Interest, capturing the road surface). It applies the **Shi-Tomasi Corner Detection** algorithm to find up to 300 unique "features"—such as tiny cracks in the asphalt, painted lane lines, gravel, or shadows. These tracked features are visualized as the **colored dots** in the demo output.

### Step 2: Temporal Tracking (Lucas-Kanade Optical Flow)
Video is processed as a sequence of discrete frames (e.g., at 30 FPS, a new frame arrives every 33 milliseconds). 

Using the **Lucas-Kanade Optical Flow** algorithm, the system takes the 300 features identified in Frame 1 and searches Frame 2 to find exactly where those specific textures shifted to. It isolates the downward vertical flow vectors (ignoring upward noise) to track the forward motion of the car.

### Step 3: Monocular Scale Conversion (Pixels to m/s)
Once the pixel displacement is tracked, the algorithm calculates the *median* pixel shift ($\Delta y$) across all surviving features. 

To convert this 2D pixel movement into a 3D physical speed, we use a Monocular Scale Approximation based on the camera's fixed height from the ground ($h$) and focal length ($f$):


$$
v_{\text{cam}} = \left( \frac{\Delta y_{median}}{\Delta t} \right) \times \left( \frac{h}{f} \right)
$$


where:
* $\Delta y_{median}$ = Median downward pixel displacement
* $\Delta t$ = Time between frames (e.g., 0.033 seconds)
* $h$ = Camera height from the road surface (e.g., 1.2 meters)
* $f$ = Assumed focal length of the smartphone camera (in pixels)

This yields an absolute forward velocity estimate ($v_{\text{cam}}$) in meters per second.

---

## 3. Sensor Fusion: Integrating into the EKF

The optical flow engine does not operate in isolation; it is deeply integrated into the navigation pipeline's **Extended Kalman Filter (EKF)**. 

When the camera successfully tracks the road surface, the resulting visual speed estimate is fed into the EKF as an observation update. The measurement matrix ($H_{vio}$) specifically isolates the velocity state variable (the 3rd state in our 4-DOF matrix):


$$
H_{vio} = \begin{bmatrix} 0 & 0 & 1 & 0 \end{bmatrix}
$$


**Why this matters:**
If the vehicle hits a massive pothole that creates chaotic IMU vibration (confusing the AI), or drives on perfectly smooth glass (starving the AI of data), the Kalman Filter seamlessly falls back to the camera's visual velocity measurement to aggressively bound longitudinal drift. This multi-sensor redundancy guarantees relentless accuracy regardless of the vehicle type or road condition.


