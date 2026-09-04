# AI-Enhanced GNSS/INS Sensor Fusion & Dead Reckoning System

An end-to-end, production-grade navigation prototype designed to maintain high-accuracy vehicle positioning during **GNSS (GPS) blackouts** (e.g., long tunnels, urban canyons, dense canopies).

The system pairs an **Edge AI & Sensor Fusion Backend** (PyTorch + Extended Kalman Filter + HMM Map-Matching) with an interactive **React Native / Leaflet Navigation Frontend** via real-time WebSockets (<100ms streaming).

---

## 🚀 Key Features & Mathematical Highlights

* **Authentic Deep Learning Speed Estimator:** A PyTorch GRU network trained on micro-vibrations & IMU dynamics from the **IO-VNBD dataset** to predict longitudinal velocity independent of satellite coverage.
* **Dynamic In-Vehicle Attitude Alignment:** Constructs a continuous 3D Direction Cosine Matrix (DCM) utilizing dynamic pitch, roll, and yaw misalignment. Automatically transforms the raw smartphone IMU feed to match the vehicle chassis regardless of arbitrary dashboard mounting angles.
* **Extended Kalman Filter (EKF) with NHC & ZUPT:** 
  * Fuses forward speed predictions and gravity-aligned gyroscope yaw.
  * **Non-Holonomic Constraints (NHC):** 1D mathematical constraint zeros out lateral sliding drift.
  * **Zero Velocity Updates (ZUPT):** Multi-factor variance gate completely halts drift when the engine is idling.
  * **Chi-Square Innovation Gating:** Statistically filters out severe multipath GNSS errors on re-acquisition.
* **HMM Viterbi Map-Matching:** Replaces naive geometric snapping with a **Hidden Markov Model**. Uses dynamic programming to compute transition penalties based on network connectivity and physical inertia, naturally locking the EKF heading to real-world topological curves using 2,065+ embedded OpenStreetMap vectors.
* **Real-Time Dual-Trajectory Visualization:** Frontend natively shows both the **Ground Truth GNSS trajectory (Green)** and the **AI Dead Reckoning fused trajectory (Blue)** side-by-side with live WebSocket telemetry metrics.

---

## 🏆 SIH Benchmark Performance

Evaluated over a **native 530-meter continuous GNSS blackout**:
* **Peak Positional Drift Metric:** 6.65%
* **Final Outage Drift Metric:** **5.89%** (Strictly passes the Smart India Hackathon <10% constraint)
* See [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) for full analytics, offline evaluation methodology, and performance plots.

---

## 🏗️ System Architecture

```
[ Smartphone 6-DOF IMU ] ──> [ 3D Attitude Alignment (DCM) ] ──> [ PyTorch GRU Speed Model ] ──┐
                                       │                                                       │
                                       ▼                                                       ▼
[ GNSS Receiver (Satellites) ] ─────────────────────────> [ Extended Kalman Filter (EKF & NHC) ]
                                                                             ▲
[ OpenStreetMap R-Tree DB ] ──> [ Viterbi HMM Decoder ] ─────────────────────┘
                                                                             │
                                                                             ▼
[ React Native / Leaflet UI ] <─── [ FastAPI WebSocket Stream (<100ms) ] <───┘
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

### Terminal 2: Start Frontend UI (Web Browser)
```bash
cd frontend
npx expo start --web
```
* The live navigation dashboard will open automatically in your browser at `http://localhost:8081`.
* **Testing on a Physical Smartphone (Expo Go):**
  1. Run `npx expo start`.
  2. Open `frontend/App.js` and update `WS_URL` to point to your computer's local Wi-Fi IP address (e.g., `ws://192.168.1.50:8000/ws`).
  3. Scan the QR code using the **Expo Go** app on Android or the Camera app on iOS.

---

## 🔬 How the Live Demonstration Works

When you observe the vehicle moving on the map:

1. **🟢 Green Mode (GNSS Active - High Accuracy):**
   * The GNSS receiver is active.
   * EKF continuously updates its position states with satellite fixes.
   * The AI trajectory (Blue line) directly overlaps the Ground Truth (Green line).

2. **🔴 Red Mode (GNSS Blackout - AI Dead Reckoning Active):**
   * A severe 500m tunnel signal loss is natively simulated.
   * GNSS is cut off completely (`gnss_status = 0`).
   * The EKF automatically relies on:
     * **PyTorch GRU Inference:** Predicts forward speed from IMU vibration signatures.
     * **Dynamic DCM Integration:** Calculates yaw rate physically free of mounting tilt errors.
     * **HMM Map Constraints:** Viterbi dynamic programming aligns the EKF steering topology directly to OpenStreetMap network curves.
   * The vehicle continues tracking the route smoothly through the outage.

3. **🔄 Seamless GNSS Recovery:**
   * When satellite reception returns, the Chi-Square Innovation Gater safely locks back onto the true coordinates with minimal glitching.
