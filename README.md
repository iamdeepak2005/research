# Proactive Congestion Avoidance in Software-Defined Networks Using Machine Learning Traffic Prediction

## Research Prototype

---

## Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [The Problem We Are Solving](#the-problem-we-are-solving)
3. [How Traditional Networks Handle Congestion](#how-traditional-networks-handle-congestion)
4. [How Our System Handles Congestion (The SDN + ML Approach)](#how-our-system-handles-congestion-the-sdn--ml-approach)
5. [System Architecture](#system-architecture)
6. [Complete System Flow (Step by Step)](#complete-system-flow-step-by-step)
7. [Project Structure](#project-structure)
8. [What Each File Does](#what-each-file-does)
9. [Technologies Used](#technologies-used)
10. [Network Topology](#network-topology)
11. [Machine Learning Pipeline](#machine-learning-pipeline)
12. [How Rerouting Works](#how-rerouting-works)
13. [Installation & Setup](#installation--setup)
14. [How to Run](#how-to-run)
15. [Understanding the Output](#understanding-the-output)
16. [How This Model Can Be Used in the Real World](#how-this-model-can-be-used-in-the-real-world)
17. [Limitations](#limitations)
18. [Future Enhancements](#future-enhancements)
19. [FAQ](#faq)

---

## What Is This Project?

This is a **research prototype** that demonstrates how a computer network can **predict traffic congestion before it happens** and **automatically reroute traffic** to avoid it.

Think of it like Google Maps for network traffic — instead of waiting for a road to become jammed and then suggesting an alternate route, our system **predicts** which road will be jammed **10 seconds in the future** and reroutes traffic **before** the jam occurs.

The system combines three technologies:
- **Software-Defined Networking (SDN)** — a modern way to control network switches using software
- **Machine Learning (ML)** — a trained model that predicts future traffic levels
- **OpenFlow Protocol** — the standard protocol for programming network switches

---

## The Problem We Are Solving

In traditional computer networks:

1. **Switches and routers make their own decisions** — each device independently decides where to send traffic
2. **No central intelligence** — no single entity sees the full picture of the network
3. **Congestion is reactive** — the network only responds AFTER congestion happens
4. **Packet loss and delays** — by the time congestion is detected, packets are already being dropped

**Result:** Users experience slow connections, video calls freeze, downloads stall, and critical services degrade.

---

## How Traditional Networks Handle Congestion

```
Traffic increases → Link becomes full → Packets start dropping
→ TCP detects loss → TCP reduces sending rate → Performance degrades
→ Eventually traffic redistributes (slowly, inefficiently)
```

This is **reactive** — the damage is already done before any correction happens.

---

## How Our System Handles Congestion (The SDN + ML Approach)

```
Controller monitors all links → ML model analyzes traffic patterns
→ Model PREDICTS: "Link X will be 90% full in 10 seconds"
→ Controller PROACTIVELY reroutes traffic to an alternate path
→ Traffic flows through the new path BEFORE congestion occurs
→ Zero packet loss, zero degradation
```

This is **proactive** — we prevent the problem before it happens.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTROL PLANE                             │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │  Telemetry   │───▶│  ML Predictor │───▶│ Routing Engine│   │
│  │  Collector   │    │  (sklearn)   │    │  (BFS paths)  │   │
│  └──────┬───────┘    └──────────────┘    └───────┬───────┘   │
│         │                                        │           │
│  ┌──────┴────────────────────────────────────────┴───────┐   │
│  │              Ryu SDN Controller                        │   │
│  │         (OpenFlow 1.3, Python-based)                   │   │
│  └──────────────────────┬────────────────────────────────┘   │
│                         │ OpenFlow Protocol                  │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────┐
│                    DATA PLANE                                │
│                         │                                    │
│     h1 ──── s1 ──────── s2 ──── h2                          │
│              │ ╲      ╱ │                                    │
│              │   ╲  ╱   │                                    │
│              │    ╳     │                                    │
│              │  ╱   ╲   │                                    │
│              │╱       ╲ │                                    │
│     h3 ──── s3 ──────── s4 ──── h4                          │
│                                                             │
│         (Mininet Virtual Network / OVS Switches)            │
└─────────────────────────────────────────────────────────────┘
```

**Control Plane** = The brain (our software — controller, predictor, routing engine)
**Data Plane** = The muscle (switches that actually forward packets)

The controller tells the switches what to do. The switches report their stats back to the controller. The controller uses ML to decide if changes are needed.

---

## Complete System Flow (Step by Step)

Here is exactly what happens, in order:

### Step 1: Network Starts
- Mininet creates 4 virtual switches (s1, s2, s3, s4) and 4 virtual hosts (h1, h2, h3, h4)
- All switches connect to the Ryu controller via OpenFlow 1.3
- The controller installs a "table-miss" rule on each switch: "if you don't know where to send a packet, ask me"

### Step 2: Basic Forwarding (Learning Switch)
- When h1 pings h2, s1 doesn't know where h2 is → sends the packet to the controller
- The controller learns: "h1 is on s1 port 1" and tells s1 where to send the packet
- Over time, the controller builds a complete MAC-to-port mapping
- Flow rules are installed so subsequent packets are forwarded directly (without asking the controller)

### Step 3: Telemetry Collection (Every 5 Seconds)
- The controller sends an `OFPPortStatsRequest` to every switch
- Each switch responds with statistics for every port:
  - `tx_bytes` — total bytes transmitted
  - `rx_bytes` — total bytes received
  - `tx_packets` — total packets transmitted
  - `rx_packets` — total packets received
  - `duration_sec` — how long the port has been active
- The controller stores these stats in:
  - **In-memory ring buffer** (TelemetryCollector — last 100 samples per port)
  - **CSV file** (DatasetLogger — permanent record for ML training)

### Step 4: Utilization Calculation
- For each port, the system calculates:
  ```
  utilization = tx_bytes / (duration_sec × link_capacity_in_bytes_per_sec)
  ```
- Link capacity = 10 Mbps = 1,250,000 bytes/sec
- A utilization of 0.80 means the link is 80% full

### Step 5: ML Model Training
- After enough data is collected (at least 30 polling cycles = 150 seconds)
- The `train_model.py` script reads the CSV file
- It engineers features from the raw data:
  - `tx_rate` — change in tx_bytes between samples
  - `rx_rate` — change in rx_bytes between samples
  - `pkt_rate` — change in packet count between samples
  - `util_avg3` — rolling average utilization over last 3 samples
  - `util_avg5` — rolling average utilization over last 5 samples
- The **target variable** is the utilization **2 samples in the future** (= 10 seconds ahead at 5-second polling)
- A Linear Regression model with StandardScaler is trained
- Model is saved as `model.joblib`

### Step 6: Real-Time Prediction
- Every 5 seconds, after receiving new stats, the `Predictor` runs
- For each (switch, port) pair:
  - Takes the last 10 telemetry records (the "window")
  - Builds a feature vector (same features as training)
  - Feeds it to the trained model
  - Gets back a predicted utilization (0.0 to 1.0)

### Step 7: Congestion Detection
- If any link's predicted utilization ≥ 0.80 (80%), it's flagged as **congested**
- The system knows WHICH specific link will be congested (e.g., s1 port 2 = the s1→s2 link)

### Step 8: Alternate Path Computation
- The `RoutingEngine` uses BFS (Breadth-First Search) on the topology graph
- It finds the shortest path from source to destination that **avoids the congested link**
- Example:
  - Default path h1→h4: `s1 → s4` (direct)
  - If s1→s2 is congested: still `s1 → s4` (doesn't use that link)
  - If s1→s4 is congested: `s1 → s3 → s4` or `s1 → s2 → s4`

### Step 9: Flow Rule Installation (Rerouting)
- The controller installs new OpenFlow flow rules on the affected switches
- These rules have **higher priority** than the default rules
- They redirect matching traffic (e.g., h1→h4) through the alternate path
- Rules have an `idle_timeout=30` — they expire after 30 seconds of no matching traffic
- This means the network automatically falls back to the default path when congestion clears

### Step 10: Continuous Monitoring
- The cycle repeats every 5 seconds
- If congestion clears, the high-priority rules expire and default forwarding resumes
- If new congestion appears, new reroute rules are installed

---

## Project Structure

```
project/
│
├── topology.py              # Mininet network topology definition
├── controller.py            # Ryu SDN controller (the brain)
├── telemetry_collector.py   # In-memory telemetry storage
├── dataset_logger.py        # CSV file logger for ML training data
├── train_model.py           # ML model training script
├── predictor.py             # Real-time congestion predictor
├── routing_engine.py        # Alternate path computation
├── utils.py                 # Shared constants and helpers
│
├── run_simulation.py        # Batch simulation (instant demo, no Mininet needed)
├── run_live.py              # Live real-time simulation (continuous, no Mininet needed)
│
├── telemetry_data.csv       # [Generated] Training data
├── model.joblib             # [Generated] Trained ML model
└── telemetry_latest.json    # [Generated] Latest telemetry snapshot
```

---

## What Each File Does

### `topology.py`
Creates the virtual network using Mininet.
- 4 OpenFlow switches (s1–s4) connected in a **full mesh** (every switch connected to every other switch)
- 4 hosts (h1–h4), one attached to each switch
- All links are 10 Mbps
- Uses a **remote controller** (connects to Ryu on port 6633)
- The full mesh provides **multiple redundant paths** between any two hosts — this is critical for rerouting to work

### `controller.py`
The Ryu SDN controller — the central brain of the system.
- Handles switch connections (OpenFlow handshake)
- Acts as a **learning switch** (learns MAC addresses, installs forwarding rules)
- **Polls port statistics** from every switch every 5 seconds
- Passes stats to the TelemetryCollector and DatasetLogger
- Calls the Predictor after each stats update
- If congestion is predicted, calls the RoutingEngine for an alternate path
- Installs high-priority flow rules to reroute traffic

### `telemetry_collector.py`
Thread-safe in-memory storage for telemetry data.
- Stores the last 100 samples per (switch, port) pair in a ring buffer
- Provides methods to get the latest record, a window of records, or all latest records
- Can export a JSON snapshot of the current network state
- Automatically computes `utilization` when storing records

### `dataset_logger.py`
Writes telemetry records to a CSV file for ML training.
- Appends one row per record (never overwrites)
- Computes the `utilization` field from tx_bytes and duration_sec
- Thread-safe (can be called from the controller's event handlers)
- Output columns: timestamp, switch_id, port_no, tx_packets, rx_packets, tx_bytes, rx_bytes, duration_sec, utilization

### `train_model.py`
Trains the machine learning model.
- Reads `telemetry_data.csv`
- Groups data by (switch_id, port_no)
- Engineers features: byte rates, packet rates, rolling averages
- Creates the target variable: utilization shifted 2 samples into the future (= 10 seconds ahead)
- Trains a scikit-learn Pipeline: StandardScaler → LinearRegression
- Reports R² score and MAE (Mean Absolute Error)
- Saves the trained model to `model.joblib`

### `predictor.py`
Loads the trained model and makes real-time predictions.
- Builds feature vectors from the telemetry window (last 10 records)
- Returns predicted utilization (0.0 to 1.0) for each link
- Flags links predicted to exceed the 80% threshold
- Has a fallback mode (returns last known utilization) if no model is available

### `routing_engine.py`
Computes alternate paths when congestion is detected.
- Maintains a graph of the switch topology
- Has a port-to-neighbor mapping (knows s1:port2 connects to s2)
- Uses BFS to find shortest path avoiding congested links
- Can list ALL available paths between any two switches
- Converts paths to flow actions (which switch should forward to which next hop)

### `utils.py`
Shared constants and helper functions used by all modules.
- `POLL_INTERVAL = 5` — seconds between stats requests
- `PREDICT_HORIZON = 10` — predict this many seconds into the future
- `UTILIZATION_THRESHOLD = 0.80` — trigger reroute above this
- `LINK_CAPACITY_BYTES = 1,250,000` — 10 Mbps in bytes/sec
- Topology maps: switch links, host-switch mapping, IP-host mapping
- Helper functions: timestamps, JSON I/O, utilization calculation

### `run_simulation.py`
A batch (static) simulation for instant demo.
- Generates 3,200 synthetic telemetry records all at once
- Trains the model
- Runs predictions on all links
- Shows rerouting decisions
- Runs in ~5 seconds, no waiting

### `run_live.py`
A live (dynamic) real-time simulation.
- Generates telemetry every 5 seconds (simulating what the Ryu controller would collect)
- Traffic patterns change over time (normal → ramp up → congestion → cooldown → random bursts)
- Auto-trains the model after 30 cycles
- Shows a live dashboard with colored progress bars
- Triggers rerouting in real-time when congestion is predicted
- Shows congestion clearing and default path restoration
- Runs continuously until Ctrl+C

---

## Technologies Used

| Technology | Purpose | Why This Choice |
|---|---|---|
| **Mininet** | Network emulator | Industry standard for SDN research; creates realistic virtual networks |
| **Ryu** | SDN controller | Python-based, well-documented, supports OpenFlow 1.3 |
| **Open vSwitch (OVS)** | Virtual switches | Most widely used software switch; full OpenFlow support |
| **OpenFlow 1.3** | Switch-controller protocol | Standard protocol for SDN; supports flow tables, group tables, meters |
| **scikit-learn** | Machine learning | Simple, reliable; Linear Regression as baseline |
| **pandas** | Data processing | Efficient CSV reading and feature engineering |
| **Python 3.10** | Programming language | Required by Ryu; good ML ecosystem |

---

## Network Topology

```
         10 Mbps        10 Mbps
  h1 ──────── s1 ─────────── s2 ──────── h2
  10.0.0.1    │ ╲           ╱ │    10.0.0.2
              │   ╲       ╱   │
              │    10 Mbps    │
              │   ╱       ╲   │
              │ ╱           ╲ │
  h3 ──────── s3 ─────────── s4 ──────── h4
  10.0.0.3         10 Mbps         10.0.0.4
```

**6 inter-switch links** (full mesh): s1-s2, s1-s3, s1-s4, s2-s3, s2-s4, s3-s4
**4 host links**: h1-s1, h2-s2, h3-s3, h4-s4
**All links**: 10 Mbps bandwidth

### Why Full Mesh?
A full mesh means there are **multiple paths** between any two hosts. For example, h1 can reach h4 via:
- s1 → s4 (direct, 1 hop)
- s1 → s2 → s4 (2 hops)
- s1 → s3 → s4 (2 hops)
- s1 → s2 → s3 → s4 (3 hops)
- s1 → s3 → s2 → s4 (3 hops)

This redundancy is essential — without multiple paths, there's nothing to reroute to.

---

## Machine Learning Pipeline

### Features (Input to the Model)

| Feature | Description |
|---|---|
| `tx_bytes` | Total bytes transmitted on this port |
| `rx_bytes` | Total bytes received on this port |
| `tx_packets` | Total packets transmitted |
| `rx_packets` | Total packets received |
| `duration_sec` | Time window of measurement |
| `utilization` | Current link utilization (0.0–1.0) |
| `tx_rate` | Change in tx_bytes since last sample |
| `rx_rate` | Change in rx_bytes since last sample |
| `pkt_rate` | Change in total packets since last sample |
| `util_avg3` | Rolling mean utilization (last 3 samples) |
| `util_avg5` | Rolling mean utilization (last 5 samples) |

### Target Variable
- **Future utilization** — the utilization value 2 samples (10 seconds) in the future
- This makes it a **regression** problem: predict a continuous value between 0.0 and 1.0

### Model
- **Algorithm**: Linear Regression (simple, interpretable baseline)
- **Preprocessing**: StandardScaler (normalizes features to zero mean, unit variance)
- **Pipeline**: StandardScaler → LinearRegression
- **Evaluation**: R² score (how well the model explains variance) and MAE (average prediction error)

### Typical Results
- **R² ≈ 0.66–0.74** — the model explains 66–74% of the variance in future utilization
- **MAE ≈ 0.05–0.07** — predictions are off by about 5–7% on average
- This is sufficient for congestion prediction (we only need to know "above 80% or not")

---

## How Rerouting Works

### Example Scenario

1. Normal state: h1 sends data to h4 via `s1 → s4` (shortest path)
2. Heavy traffic starts flowing on the s1→s2 link (port 2 of s1)
3. ML model predicts: "s1:port2 will be at 92% utilization in 10 seconds"
4. 92% > 80% threshold → **congestion alert!**
5. Routing engine computes alternate path **avoiding the s1→s2 link**
6. Since h1→h4 uses `s1 → s4` (not s1→s2), the current path is fine
7. But if h1→h2 traffic was the problem, it would reroute via `s1 → s3 → s2`

### OpenFlow Rules Installed

```
Match:  eth_type=IPv4, src=10.0.0.1, dst=10.0.0.4
Action: output to port [alternate path port]
Priority: 10 (higher than default priority 1)
Idle timeout: 30 seconds (auto-expires when traffic stops)
```

---

## Installation & Setup

### Option A: Windows (Simulation Only — No Real Network)

```bash
cd project
python -m venv venv
.\venv\Scripts\pip install scikit-learn pandas numpy joblib
```

### Option B: Ubuntu 22.04 (Full System with Mininet)

```bash
# Install system packages
sudo apt update
sudo apt install mininet openvswitch-switch python3-pip python3-venv

# Create virtual environment
cd project
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install ryu eventlet scikit-learn pandas numpy joblib
```

---

## How to Run

### Quick Demo (Windows or Linux — No Mininet Needed)

**Batch mode** (runs in ~5 seconds, shows all results at once):
```bash
python run_simulation.py
```

**Live mode** (runs continuously, real-time dashboard):
```bash
python run_live.py
# Press Ctrl+C to stop
```

### Full System (Ubuntu with Mininet)

**Terminal 1 — Start the SDN controller:**
```bash
cd project
source venv/bin/activate
ryu-manager controller.py
```

**Terminal 2 — Start the virtual network:**
```bash
cd project
sudo python3 topology.py
```

**Terminal 3 (inside Mininet CLI) — Generate traffic:**
```bash
# Test connectivity
mininet> pingall

# Start iperf server on h1
mininet> h1 iperf -s &

# Send heavy traffic from h4 to h1 for 2 minutes
mininet> h4 iperf -c 10.0.0.1 -t 120 -b 9M
```

**After ~60 seconds of traffic — Train the model:**
```bash
python3 train_model.py
```

**Restart the controller to use the trained model:**
```bash
ryu-manager controller.py
```

Now generate heavy traffic again — the controller will predict congestion and reroute automatically.

---

## Understanding the Output

### run_live.py Dashboard

```
─────────────────────────────────────────────────────────────────
  Cycle 31  │  Time: 152s  │  Model: trained (R²)  │  Reroute: ACTIVE
─────────────────────────────────────────────────────────────────
  s1:p2  ███████████████████████░░  95.1% ⚠ CONGESTED
  s1:p3  ████░░░░░░░░░░░░░░░░░░░░░  17.9%
  s1:p4  ███░░░░░░░░░░░░░░░░░░░░░░  14.9%
  s2:p3  ███████████░░░░░░░░░░░░░░  44.2%
  other  avg=15.1%  max=20.4%  (12 links)
```

- **Cycle**: How many polling rounds have completed
- **Time**: Seconds since simulation started
- **Model**: Whether ML model is trained
- **Reroute**: Whether an alternate path is currently active
- **Progress bars**: Green = OK, Yellow = moderate, Red = congested
- **⚠ CONGESTED**: This link's predicted utilization exceeds 80%

### train_model.py Output

```
[INFO] Loaded 3200 rows from telemetry_data.csv
[INFO] After feature engineering: 3168 rows
[RESULT] R² = 0.7117
[RESULT] MAE = 0.061038
[INFO] Model saved to model.joblib
```

- **R²** (R-squared): 1.0 = perfect prediction, 0.0 = random guess. 0.71 is good for a simple linear model.
- **MAE** (Mean Absolute Error): Average error in utilization units. 0.06 means predictions are off by ~6%.

---

## How This Model Can Be Used in the Real World

### 1. University/Campus Networks
- Deploy OVS switches on network servers
- Connect them to a Ryu/ONOS controller
- Collect real traffic data and train the model
- Automatically balance load across redundant links

### 2. Data Center Networks
- Data centers already use SDN (Google, Facebook, Microsoft)
- This system can predict hotspots in the network fabric
- Proactively redistribute VM traffic before links saturate

### 3. ISP/Telecom Networks
- Internet Service Providers can use SDN for traffic engineering
- Predict peak-hour congestion and reroute customer traffic
- Reduce packet loss and improve QoS (Quality of Service)

### 4. Research & Education
- Demonstrate SDN concepts in networking courses
- Benchmark different ML algorithms for traffic prediction
- Publish papers comparing reactive vs. proactive congestion management
- Extend to larger topologies (fat-tree, leaf-spine)

### 5. Real Deployment Requirements
To move from this prototype to a real deployment, you would need:
- **Real SDN switches** (hardware: HP, Cisco, Arista with OpenFlow support; or software: OVS on Linux servers)
- **A production controller** (ONOS or OpenDaylight instead of Ryu for scale)
- **Real traffic data** for training (not synthetic)
- **More sophisticated ML** (LSTM, Transformer models for time-series)
- **High availability** (redundant controllers, failover)

---

## Limitations

1. **Simulated environment** — This runs on Mininet (virtual network), not a real physical network
2. **Linear Regression** — Simple model; may not capture complex traffic patterns
3. **Static topology** — The switch graph is hardcoded; doesn't auto-discover topology changes
4. **Single flow rerouting** — Currently reroutes h1→h4 traffic as a demo; production systems need per-flow granularity
5. **No QoS metrics** — Doesn't consider latency or jitter, only bandwidth utilization
6. **10 Mbps links** — Toy bandwidth; real networks are 1–100 Gbps

---

## Future Enhancements

| Enhancement | Difficulty | Impact |
|---|---|---|
| Replace Linear Regression with **LSTM** (deep learning) | Medium | Better time-series prediction |
| Add **latency prediction** (not just bandwidth) | Medium | More complete congestion picture |
| **Auto-discover topology** using LLDP | Medium | No hardcoded switch graph |
| Support **larger topologies** (fat-tree, 16+ switches) | Easy | More realistic scenarios |
| Add a **web dashboard** (Flask/React) | Medium | Visual monitoring |
| **Per-flow rerouting** (not just h1→h4) | Hard | Production-ready routing |
| **Multi-controller** setup | Hard | Scalability |
| Export metrics to **Grafana/Prometheus** | Easy | Professional monitoring |
| Compare **multiple ML models** (RF, SVM, XGBoost, LSTM) | Easy | Research contribution |

---

## FAQ

### Is this monitoring my real Wi-Fi / home network?
**No.** This creates a completely separate virtual network using Mininet. Your real network is not affected in any way.

### Do I need special hardware?
**No.** Everything runs in software on a single machine. Mininet creates virtual switches and hosts.

### Can I use this for my thesis/research paper?
**Yes.** This is designed as a research prototype. You can collect data, generate graphs, compare ML models, and present results.

### Why Ryu and not ONOS/OpenDaylight?
Ryu is Python-based (easy to integrate with scikit-learn), lightweight, and well-documented. For production, ONOS or OpenDaylight would be better.

### Why Linear Regression and not Deep Learning?
We start simple. Linear Regression is interpretable and fast. The code is modular — you can swap in LSTM or any other model by modifying `train_model.py` and `predictor.py`.

### What does R² = 0.71 mean?
The model explains 71% of the variance in future utilization. For a simple linear model on synthetic data, this is reasonable. With real network data and better models (LSTM), this would improve.

### How accurate is the 10-second prediction?
With MAE ≈ 0.06, predictions are off by about 6% on average. This is accurate enough for congestion prediction — we only need to distinguish "above 80%" from "below 80%".

---

## License

This is a research prototype for educational and academic purposes.

---

## Author

Research Prototype — Proactive Congestion Avoidance in SDN using ML Traffic Prediction
