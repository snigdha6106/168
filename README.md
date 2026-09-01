# Intelligent Dead Reckoning (IDR) System with GNSS Fusion

This project is a complete end-to-end working prototype for an AI-enhanced Dead Reckoning and Sensor Fusion system, designed to handle GNSS (GPS) blackouts in environments like tunnels, urban canyons, or dense forests.

It consists of a Python-based **Edge Software Engine** that runs advanced Deep Learning and Sensor Fusion algorithms, and a **React Native Mobile Application** that provides a real-time navigation interface.

## System Architecture

The prototype is divided into two main components:

1. **`backend/` (Edge Software Engine)**
   * **Dataset Generator**: Simulates a vehicle's trajectory, generating noisy IMU data (accelerometer/gyroscope) and simulated GNSS dropouts, mimicking the IO-VNBD dataset structure.
   * **AI Speed & Vibration Filter**: A PyTorch GRU Neural Network that filters out IMU noise and predicts the vehicle's forward velocity.
   * **GNSS+INS Sensor Fusion**: An Extended Kalman Filter (EKF) that fuses GNSS data with the AI-predicted IMU odometry. It includes a simulated Map-Matching Non-Holonomic Constraint (NHC) to prevent heading drift.
   * **Simulation Server**: A FastAPI WebSocket server that streams live telemetry.

2. **`frontend/` (Real-time Navigation Interface)**
   * Built with React Native (Expo).
   * Features a live interactive map (`react-leaflet` for web fallback, `react-native-maps` for native).
   * Displays real-time velocity and active tracking mode (GNSS vs AI Dead Reckoning).

---

## Installation & Setup

### Prerequisites
* Python 3.9+
* Node.js & npm
* Expo CLI

### 1. Setup the Backend (Edge Engine)
Navigate to the backend directory, create a virtual environment, and install the dependencies:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn websockets pandas numpy torch filterpy
```

### 2. Setup the Frontend (Mobile App)
Navigate to the frontend directory and install the Node modules:

```bash
cd frontend
npm install
```

---

## How to Run the Prototype

To see the live simulation, you must run both the backend and frontend simultaneously in separate terminal windows.

### Start the Backend
```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```
*Note: On first startup, the backend will automatically generate the synthetic dataset and train the PyTorch AI model. This may take a few seconds.*

### Start the Frontend
Open a new terminal window:
```bash
cd frontend
npx expo start --web
```
* This will open the real-time dashboard in your web browser. 
* To view the app on a physical mobile device, simply run `npx expo start`, download the **Expo Go** app on your phone, and scan the QR code. *(Ensure `WS_URL` in `App.js` is set to your computer's local IP address).*

---

## Demonstration Highlights

When you run the application, you will observe a vehicle navigating a route:
1. **Green Mode (GNSS+INS Fusion)**: The system relies on satellites. The AI's estimated path (blue line) perfectly overlays the ground truth (green line).
2. **Red Mode (AI Dead Reckoning)**: A tunnel is simulated, and GNSS drops out completely. The system instantly falls back to the PyTorch AI model analyzing simulated smartphone vibrations. The blue line continues to accurately track the route without GPS!
3. **Seamless Transition**: When the vehicle exits the "tunnel", GNSS is restored. Notice that because the AI model is highly accurate, there is almost zero jump or skipping when the signal reconnects.
