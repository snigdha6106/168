# AI-Enhanced GNSS/INS Sensor Fusion & Dead Reckoning System

An end-to-end, production-grade navigation prototype designed to maintain high-accuracy vehicle positioning during **GNSS (GPS) blackouts** (e.g., long tunnels, urban canyons, dense canopies).

The system pairs an **Edge AI & Sensor Fusion Backend** (PyTorch + Extended Kalman Filter + OpenStreetMap Map-Matching) with an interactive **React Native / Leaflet Navigation Frontend** via real-time WebSockets (<100ms streaming).

---

## 🚀 Key Features & Highlights

* **Authentic Deep Learning Speed Estimator:** A PyTorch GRU network trained on micro-vibrations & IMU dynamics from the **IO-VNBD dataset** (`S-S2.csv`) to predict longitudinal velocity without data leakage or overfitting on test routes (`S-S1.csv`).
* **3D Gravity Vector Decoupling:** Uses the device's 3-axis gravity vector to project raw gyroscope rates onto the true vertical axis, ensuring orientation-independent yaw tracking regardless of smartphone mounting angle.
* **Extended Kalman Filter (EKF):** Fuses forward speed predictions, integrated gyroscope yaw, and GNSS observations with dynamic covariance adaptation during satellite outages.
* **Real OpenStreetMap (OSM) Spatial Indexing:** Uses Shapely's `STRtree` R-Tree spatial indexing loaded with **2,065+ real OpenStreetMap road vector segments** (Coventry, UK). Performs non-holonomic projection and bidirectional road tangent alignment without heading flips.
* **Real-Time Dual-Trajectory Visualization:** Frontend shows both the **Ground Truth GNSS trajectory (Green)** and the **AI Dead Reckoning fused trajectory (Blue)** side-by-side with live status indicators.

---

## 🏗️ System Architecture

```
[ Smartphone 6-DOF IMU ] ──> [ 3D Gravity Vector Alignment ] ──> [ PyTorch GRU Speed Model ] ──┐
                                       │                                                      │
                                       ▼                                                      ▼
[ GNSS Receiver (Satellites) ] ───────────────────────────────> [ Extended Kalman Filter (EKF) ]
                                                                             ▲
[ OpenStreetMap R-Tree DB ] ──> [ Non-Holonomic Tangent Snap ] ──────────────┘
                                                                             │
                                                                             ▼
[ React Native / Leaflet UI ] <─── [ FastAPI WebSocket Stream (<100ms) ] <────┘
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
> **Note:** The backend automatically loads the pre-cached `data/osm_roads.json` (2,065 OSM segments) and pre-trained weights (`model_weights.pth`).

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
   * A tunnel or severe signal loss is simulated for 30% of the route.
   * GNSS is cut off completely (`gnss_status = 0`).
   * The EKF automatically relies on:
     * **PyTorch GRU Inference:** Predicts forward speed from IMU vibration signatures.
     * **Gravity-Aligned Gyroscope Integration:** Calculates yaw rate free of tilt error.
     * **OSM Map Constraints:** Snaps position to real road geometry using R-Tree spatial indexing.
   * The vehicle continues tracking the route smoothly through the outage.

3. **🔄 Seamless GNSS Recovery:**
   * When satellite reception returns, EKF smoothly re-converges with zero jumping or glitching.

---

## 📁 Repository Structure

```
SIH26/
├── README.md                          # Project documentation & instructions
├── backend/                           # Edge AI & Sensor Fusion Engine
│   ├── main.py                        # FastAPI WebSocket telemetry server
│   ├── sensor_fusion.py               # EKF + 3D Gravity Alignment + OSM Map-Matching
│   ├── ai_model.py                    # PyTorch GRU Neural Network architecture
│   ├── parse_iovnbd.py                # Dataset parsing & preprocessing script
│   ├── model_weights.pth              # Pre-trained GRU model weights
│   └── data/
│       ├── osm_roads.json             # 2,065 OpenStreetMap road segments (Coventry, UK)
│       ├── real_route_processed.csv   # S-S1 Test Route (with simulated tunnel)
│       └── train_route_processed.csv  # S-S2 Training Route (zero leakage)
└── frontend/                          # Mobile / Web Telemetry Dashboard
    ├── App.js                         # Main dashboard & telemetry handler
    ├── WebMap.js                      # Leaflet interactive map component
    ├── package.json                   # Node dependencies & Expo configuration
    └── assets/                        # Icons and application branding
```
