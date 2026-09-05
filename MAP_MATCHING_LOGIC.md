# Offline Map-Matching & Spatial Indexing Logic

When the vehicle enters a GNSS-denied zone, dead-reckoning inherently accumulates drift. Map-matching snaps the estimated trajectory to the known road topology to bound this error. This document explains the spatial search architecture and the Hidden Markov Model (HMM) logic used to achieve this in real-time.

## 1. Spatial Data & Search Structure

### What Spatial Indexing Solves
In dead reckoning, the vehicle produces an updated coordinate (lat, lon) at **10 Hz** (every 100 ms). A regional road network typically contains tens of thousands of road segments (e.g., OpenStreetMap extract).
* **The Naive Approach ($O(N)$ Linear Scan):** At every 100 ms frame, calculating the distance to *every single road segment* requires hundreds of thousands of complex geometric projections per second. On an edge device, this causes severe execution lag.
* **The Spatial Indexing Approach ($O(\log N)$):** A spatial index organizes geometric shapes hierarchically using bounding boxes. It discards 99.9% of the map in a few operations, retrieving only the 3 to 5 road segments within a **30-meter radius** of the car.

### Storage Technology
Roads are extracted from OpenStreetMap (OSM) geometries and stored as coordinate lines in a local static JSON file (`data/osm_roads.json`). No network connection or map tile server is required during runtime.

### How an R-Tree Works
An **R-Tree** (Rectangle Tree) indexes multi-dimensional information:
1. **Minimum Bounding Boxes (MBR):** Every road segment is enclosed in its smallest possible axis-aligned rectangle:
   

$$
\text{MBR} = [lon_{\min}, lat_{\min}, lon_{\max}, lat_{\max}]
$$

2. **Hierarchical Grouping:** Nearby road segments' bounding boxes are grouped into larger parent boxes recursively, forming a balanced tree.
3. **Fast Spatial Pruning:** When searching for roads within 30 meters, the algorithm checks the root node. If a high-level bounding box does not intersect the search radius, every road inside it is instantly eliminated.

### What is an STRtree (Sort-Tile-Recursive Tree)?
Because a road network is static, we optimize the R-Tree upfront using **bulk loading** (STRtree):
1. **Sort:** Road segment centroids are sorted along the X-axis (longitude).
2. **Tile:** Partitioned into vertical slices.
3. **Sort & Tile Again:** Sorted along the Y-axis (latitude) and grouped.
This maximizes node packing and eliminates empty space inside bounding boxes, making candidate road retrieval nearly instantaneous ($< 0.1$ ms).

#### Comparison: R-Tree vs. KD-Tree
| Feature | R-Tree / STRtree | KD-Tree |
| --- | --- | --- |
| **Best Used For** | **Line segments and polygons** (roads, buildings) | **Zero-dimensional points** (GPS coordinates) |
| **Handling Geometry** | Native. An MBR wraps the full length of a road. | Awkward. Roads must be chopped into dense points. |

## 2. Mathematical Formulation: HMM & Viterbi Decoding

Once candidate roads $C_t = \{c_t^1, c_t^2, \dots, c_t^K\}$ are retrieved via the STRtree, the algorithm models the true vehicle road state as a hidden sequence $R_{1:T}$ and the dead-reckoning outputs as observation states $Z_{1:T}$.

### 1. Emission Probability ($P(z_t \mid r_t)$)
For candidate road $r_t^{(i)}$, the algorithm computes the perpendicular Euclidean projection distance $d_i$ from the dead-reckoning coordinate to the road centerline. A zero-mean Gaussian calculates the likelihood:

$$
\log P(z_t \mid r_t^{(i)}) = -\frac{1}{2}\left(\frac{d_i}{\sigma_z}\right)^2
$$

where $\sigma_z \approx 10.0\text{ m}$. Greater distances incur quadratic penalties.

### 2. Transition Probability ($P(r_t^{(j)} \mid r_{t-1}^{(i)})$)
Evaluates the topological plausibility of moving from road segment $r_{t-1}^{(i)}$ to $r_t^{(j)}$ over $\Delta t$. It compares dead-reckoning displacement $\Delta d_{\text{DR}} = \Vert z_t - z_{t-1} \Vert$ with along-network driving distance $\Delta d_{\text{network}}$:

$$
\log P(r_t^{(j)} \mid r_{t-1}^{(i)}) = -\frac{\vert \Delta d_{\text{DR}} - \Delta d_{\text{network}} \vert}{\beta} - \lambda \cdot \mathbb{I}_{\text{discontinuous}}
$$

where $\beta \approx 5.0\text{ m}$. $\mathbb{I}_{\text{discontinuous}}$ heavily penalizes jumping to topologically disconnected roads.

### 3. Viterbi Dynamic Programming Loop
Computes the cumulative path probability recursively:

$$
V_t(j) = \max_i \left( V_{t-1}(i) + \log P(r_t^{(j)} \mid r_{t-1}^{(i)}) \right) + \log P(z_t \mid r_t^{(j)})
$$

* **Underflow Prevention:** Probabilities are normalized at each step: $V_t(j) \leftarrow V_t(j) - \max_k V_t(k)$.
* The state maximizing $V_t$ selects the active road. The dead-reckoning position is snapped onto the centerline, feeding back into the EKF to bind lateral drift.
