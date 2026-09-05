# GNSS-Denied Kinematic Updates & Integration

When GNSS signals are lost, the application relies entirely on self-contained inertial sensors and AI to compute the continuous coordinate updates. This document details the live data ingestion, dynamic alignment, and Extended Kalman Filter (EKF) integration.

## 1. Live IMU Data Ingestion from the Phone

### Sensor Acquisition Architecture
Hardware motion capture utilizes the **`expo-sensors`** API running inside the React Native client runtime (`frontend/App.js`).

```javascript
import { Accelerometer, Gyroscope } from 'expo-sensors';

// Lock sampling loop to 10 Hz (100 ms intervals)
Accelerometer.setUpdateInterval(100);
Gyroscope.setUpdateInterval(100);

Accelerometer.addListener(accData => { /* { x, y, z } in Gs */ });
Gyroscope.addListener(gyroData => { /* { x, y, z } in rad/s */ });
```

Incoming readings are converted to SI units ($\text{m/s}^2$) and appended to an in-memory sliding window (e.g., $W = 10$ frames representing 1.0 s). High-frequency components represent chassis motion; low-pass filtering isolates the dynamic gravity vector $g = [g_x, g_y, g_z]^T$.

### In-Vehicle Dynamic Attitude Alignment (DCM)
Before inference, the phone's arbitrary mount angles are realigned to the car's axes:
1. **Tilt Leveling:** Pitch ($\theta$) and roll ($\phi$) angles are derived from the gravity vector:
   

$$
\phi = \text{atan2}(g_y, g_z), \quad \theta = \text{atan2}\left(-g_x, \sqrt{g_y^2 + g_z^2}\right)
$$

2. **Yaw Alignment ($\psi$):** During confirmed forward acceleration with active GNSS ($v > 3\text{ m/s}$), the horizontal acceleration identifies the chassis heading, updated via a low-pass filter:
   

$$
\psi \leftarrow \psi + \alpha \cdot (\text{atan2}(a_{y,\text{level}}, a_{x,\text{level}}) - \psi)
$$

3. **Direction Cosine Matrix Transformation:**
   

$$
R_b^v = R_z(-\psi) R_y(\theta) R_x(\phi)
$$

   

$$
a_{\text{aligned}} = R_b^v \cdot a_{\text{raw}}, \quad \omega_{\text{aligned}} = R_b^v \cdot \omega_{\text{raw}}
$$

This mathematically levels and rotates raw 6-DoF inputs into the vehicle chassis frame, completely isolating the integration from arbitrary dashboard mount angles.

## 2. Calculating the Updated Position

### AI Velocity Inference
Instead of numerically integrating $a_x$ (which causes catastrophic $O(t^2)$ error propagation), the sliding IMU window feeds into the temporal AI model. The forward velocity $v_{\text{pred}}$ is inferred directly from inertial vibration profiles. A momentum blending filter applies dynamic inertia:

$$
v_t = 0.3 \cdot v_{t-1} + 0.7 \cdot v_{\text{pred}}
$$

### Discrete Extended Kalman Filter (EKF) Propagation
The system maintains a 4-DOF state vector $x$:

$$
x = \begin{bmatrix} \text{lat} & \text{lon} & v & \theta \end{bmatrix}^T
$$

where $\text{lat}, \text{lon}$ are coordinates in degrees, $v$ is longitudinal velocity (m/s), and $\theta$ is heading angle (rad).

**Kinematic Process Update ($\Delta t = 0.1\text{ s}$):**

$$
\theta_t = \theta_{t-1} + \omega_{z,\text{aligned}} \cdot \Delta t
$$

$$
\text{lat}_t = \text{lat}_{t-1} + \frac{v_t \cdot \cos(\theta_t) \cdot \Delta t}{M_{\text{lat}}}
$$

$$
\text{lon}_t = \text{lon}_{t-1} + \frac{v_t \cdot \sin(\theta_t) \cdot \Delta t}{M_{\text{lon}}(\text{lat})}
$$

where the Earth radius conversions are:

$$
M_{\text{lat}} = 111320.0\text{ m/deg}, \quad M_{\text{lon}} = 111320.0 \cdot \cos(\text{lat})\text{ m/deg}
$$

### Filter Hardening Constraints
Constraints are applied at every prediction step to halt accumulation of drift during challenging dynamics:
* **ZUPT / ZARU (Zero Velocity/Angular Rate Update):** If sliding window acceleration variance drops below a threshold ($\text{Var}(\Vert a \Vert) < 0.02\text{ m/s}^2$), vehicle stopped conditions are rigorously enforced: $v_t = 0$, $\omega_{z,t} = 0$.
* **1D Non-Holonomic Constraint (NHC):** Enforces non-slip lateral dynamics ($v_{\text{lateral}} \approx 0$) via a pseudo-measurement matrix $H = \begin{bmatrix} 0 & 0 & 0 & v \end{bmatrix}$, eliminating sideways inertial drift before it integrates into the positional state.
