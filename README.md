# AI-Enhanced GNSS/INS Sensor Fusion & Dead Reckoning System

An end-to-end, production-grade navigation prototype designed to maintain high-accuracy vehicle positioning during **GNSS (GPS) blackouts** (e.g., long tunnels, urban canyons, dense canopies).

The system pairs a powerful **Python Edge AI & Sensor Fusion Backend** (PyTorch + Extended Kalman Filter + HMM Map-Matching) with an interactive **React Native / Leaflet Navigation Frontend** via real-time WebSockets (<100ms streaming). 

It also ships with a **Zero-Dependency On-Device Edge Inference Engine** built directly into the React Native frontend, enabling true offline autonomy without CocoaPods, Native Modules, or network connectivity!

---

## 🚀 Key Features & Mathematical Highlights

### Core Backend (Sensor Fusion & Map Matching)
* **Authentic Deep Learning Speed Estimator:** A PyTorch GRU network trained on micro-vibrations & IMU dynamics from the **IO-VNBD dataset** to predict longitudinal velocity independent of satellite coverage.
* **Dynamic In-Vehicle Attitude Alignment:** Constructs a continuous 3D Direction Cosine Matrix (DCM) utilizing dynamic pitch, roll, and yaw misalignment. Automatically transforms the raw smartphone IMU feed to match the vehicle chassis regardless of arbitrary dashboard mounting angles.
* **Extended Kalman Filter (EKF) with NHC & ZUPT:** 
  * Fuses forward speed predictions and gravity-aligned gyroscope yaw.
  * **Non-Holonomic Constraints (NHC):** 1D mathematical constraint zeros out lateral sliding drift.
  * **Zero Velocity Updates (ZUPT):** Multi-factor variance gate completely halts drift when the engine is idling.
  * **Chi-Square Innovation Gating:** Statistically filters out severe multipath GNSS errors on re-acquisition.
* **HMM Viterbi Map-Matching:** Replaces naive geometric snapping with a **Hidden Markov Model**. Uses dynamic programming to compute transition penalties based on network connectivity and physical inertia, naturally locking the EKF heading to real-world topological curves using 2,065+ embedded OpenStreetMap vectors.

### Core Frontend (Edge AI & Live Hardware Testing)
* **Zero-Dependency Edge AI (Pure JS):** The PyTorch LSTM architecture is explicitly reconstructed using pure JavaScript matrix operations (Sigmoid/Tanh/Temporal Gates). This allows neural network velocity predictions to execute locally on the iPhone CPU inside standard Expo Go.
* **Live Sensor Hardware Mode:** Utilizes `expo-sensors` and `expo-location` to feed the phone's *physical* gyroscope and accelerometer data directly into the Edge AI engine while locking onto the user's real-world GPS coordinates. 
* **Standalone Offline Replay:** Packaged with a localized JSON stream of the dataset, allowing the vehicle map to simulate dead-reckoning completely offline even if the Python backend is shut down.
* **Interactive Free-Roam Map:** Intelligent Pan/Zoom overrides dynamically release camera locks when the user touches the screen, combined with a "Recenter Camera" button to resume telemetry tracking.

---


## 🏆 SIH Benchmark Performance

Evaluated over a **native 530-meter continuous GNSS blackout**:
* **Peak Positional Drift Metric:** 6.65%
* **Final Outage Drift Metric:** **5.14%** (Strictly passes the Smart India Hackathon <10% constraint)

### Component Accuracy Achieved
To prove the rigorous mathematical contribution of our components, we measured their specific impacts during the 530m outage:
* **IMU / AI Data Accuracy:** Unassisted dead-reckoning (Raw IMU) results in a catastrophic **250.09%** drift due to $O(t^2)$ exponential integration error. By injecting our LSTM AI velocity predictions, applying Dynamic Attitude Alignment, and enforcing Non-Holonomic Constraints (NHC) in the EKF, the pure inertial drift is dramatically bounded to **80.33%**.
* **Map-Matching Accuracy:** While the AI accurately tracks forward distance, minor gyroscope noise causes lateral drift. By applying spatial indexing (STRtree) and topological transition constraints via a Hidden Markov Model (HMM), the map-matcher perfectly binds the vehicle to the road geometry. This topological correction truncates the remaining 80.33% error down to the final pinpoint accuracy of **5.14%**.

*See [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) and [`SYSTEM_ACCURACY_ANALYSIS.md`](SYSTEM_ACCURACY_ANALYSIS.md) for full analytics, offline evaluation methodology, and performance plots.*

---
---

## 🏗️ System Architecture

```text
[ Server Backend ] 
[ PyTorch GRU Speed Model ] <── [ 3D Attitude Alignment (DCM) ] <── [ Dataset IMU ]
           │                                                                
           ▼                                                                
[ Extended Kalman Filter ] <─── [ Viterbi HMM Decoder ] <──────────── [ OSM R-Tree DB ]
           │
           ▼
[ FastAPI WebSocket Stream ]
           │
           ▼
[ React Native App (Client) ] ─── (Cloud/Server AI Mode) ──> [ UI Dashboard & Map ]
                                                                      ▲
[ Expo Hardware Sensors ] ─── (On-Device Edge AI Mode) ───────────────┤
                                      │                               │
                                      ▼                               │
                          [ Pure JS LSTM Inference ] ─────────────────┘
```

---

## 📋 Prerequisites

* **Python:** Version `3.10` or higher
* **Node.js:** Version `18.x` or higher & `npm`
* **Expo CLI:** Installed via `npm` or run via `npx`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SIH26
```

### 2. Backend Setup
Navigate into the `backend/` folder, create a virtual environment, and install dependencies:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install fastapi uvicorn websockets pandas numpy torch filterpy shapely
```

### 3. Frontend Setup
Open a new terminal window, navigate into the `frontend/` folder, and install npm packages:

```bash
cd frontend
npm install
```

---

## ▶️ Running the Application

To run the live simulation, start both the **Backend** and **Frontend** concurrently in separate terminals.

### Terminal 1: Start Backend Engine
```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Terminal 2: Start Frontend UI (Physical iPhone or Web)
```bash
cd frontend
npx expo start --host lan
```
* **Testing on a Physical Smartphone (Recommended):**
  1. Ensure your phone and computer are on the same Wi-Fi network.
  2. The `frontend/App.js` automatically routes WebSockets to the computer's Local IP (`WS_URL = 'ws://<YOUR-IP>:8000/ws/telemetry'`).
  3. Scan the QR code in the terminal using the **Expo Go** app on Android or the Camera app on iOS.

---

## 🔬 Interactive Dashboard Modes

The React Native application features two powerful toggle switches for SIH presentation:

1. **[🧠 On-Device Edge AI] vs [☁️ Cloud / Server AI]**
   * **Cloud AI:** Standard mode. The PyTorch backend streams down the fused EKF telemetry.
   * **Edge AI:** Kills the network dependency. The React Native app natively intercepts the raw IMU sensor stream and runs the neural network mathematically on the iPhone CPU. 

2. **[📡 Live Device Hardware] vs [📼 Dataset Benchmark]**
   * **Dataset Benchmark:** Replays the official IO-VNBD UK dataset route. If "Edge AI" is active while the backend is dead, it falls back to a standalone offline JSON replay proving true offline autonomy.
   * **Live Device Hardware:** Grants the app physical permissions. The map centers on your real physical location in India, and the AI engine actively calculates velocity using your phone's live `expo-sensors` hardware!
