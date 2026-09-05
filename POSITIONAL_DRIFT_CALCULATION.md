# Positional Drift Calculation & Benchmarking

The evaluation script (`evaluate_benchmarks.py`) runs the unassisted dead-reckoning pipeline across the native 530.30 m GNSS blackout zone in the IO-VNBD dataset. This document details how the final SIH evaluation metrics are computed.

## Distance Calculations via Haversine Metric
To compute the true physical distance over the Earth's surface between any two coordinates $(\phi_1, \lambda_1)$ and $(\phi_2, \lambda_2)$, the Haversine formula is used:

$$
a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)
$$

$$
d(\mathbf{x}_1, \mathbf{x}_2) = 2 R \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right)
$$

where $R = 6371000\text{ m}$ is the Earth's radius, and $\Delta\phi, \Delta\lambda$ are differences in latitude and longitude (in radians).

## Algorithmic Step-by-Step Execution During Outage

### 1. Accumulated Distance Traveled ($D_{blackout}$)
At each frame $k$ where $\text{gnss-status} = 0$, the physical displacement is integrated from successive ground truth coordinates ($\mathbf{x}_{GT, k-1}$ to $\mathbf{x}_{GT, k}$):

$$
D_{blackout} = \sum_{k \in \text{outage}} d(\mathbf{x}_{GT, k-1}, \mathbf{x}_{GT, k})
$$

For the native benchmark zone, this total accumulation equals **530.30 m**.

### 2. Instantaneous Euclidean Position Error ($E_{pos}$)
At every frame, the absolute deviation between the system's estimated coordinate $\mathbf{x}_{EKF, k}$ and the hardware RTK ground truth $\mathbf{x}_{GT, k}$ is measured:

$$
E_{pos}(k) = d(\mathbf{x}_{GT, k}, \mathbf{x}_{EKF, k})
$$

### 3. Cumulative Drift Percentage
To evaluate the system's performance against the 10% Smart India Hackathon target constraint, the cumulative percentage is computed continuously for every step where $D_{blackout} > 20\text{ m}$:

$$
\text{Drift}(k) = \left( \frac{E_{pos}(k)}{D_{blackout}(k)} \right) \times 100
$$

### 4. Final Exit Benchmark Metric
The most critical metric is the final deviation exactly at the moment GNSS is reacquired. At the final frame of the outage (frame 1400/1401), the final positional error is extracted. 

For example, if the final positional error $E_{pos} = 34.14\text{ m}$:

$$
\text{Final Drift} = \left( \frac{34.14\text{ m}}{530.30\text{ m}} \right) \times 100 = \mathbf{6.44}
$$

Because this metric evaluates to $< 10.0$, the pipeline officially passes the ISRO problem statement benchmark criteria.
