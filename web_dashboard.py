#!/usr/bin/env python3
"""
web_dashboard.py — Web-based real-time dashboard with network topology.

Serves a single-page HTML dashboard at http://localhost:8050
with a JSON API endpoint at /api/data for live data updates.

Features:
  - Animated network topology with switches, hosts, links
  - Packet flow animation along active path
  - Congestion detection & reroute visualization
  - Real-time utilization charts (Chart.js)
  - Link gauges, ML metrics, event log
  - Dynamic topology (supports N switches)

Uses ONLY Python built-ins (http.server, json, threading).
Frontend uses Chart.js from CDN for smooth animated graphs.
"""

import json
import threading
import time
import webbrowser
import collections
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

MAX_HISTORY = 200


class DashboardState:
    """Thread-safe shared state between the engine and the HTTP server."""

    def __init__(self, n_switches=4):
        self._lock = threading.Lock()
        self._n_switches = n_switches
        self._cycle = 0
        self._elapsed = 0.0
        self._mode = "simulate"
        self._model_status = "collecting (0/30)"
        self._reroute_active = False
        self._reroute_path = ""
        self._train_time_ms = 0
        self._predict_time_ms = 0
        self._r2_score = None
        self._mae_score = None

        # Time-series
        self._times = collections.deque(maxlen=MAX_HISTORY)
        self._util_history = collections.defaultdict(
            lambda: collections.deque(maxlen=MAX_HISTORY)
        )
        self._nic_tx = collections.deque(maxlen=MAX_HISTORY)
        self._nic_rx = collections.deque(maxlen=MAX_HISTORY)

        # All predictions (latest)
        self._predictions = {}
        self._congested = []

        # Topology: active path, default path
        self._default_path = []
        self._active_path = []
        self._device_info = None  # discovered device info

        # Events
        self._events = collections.deque(maxlen=50)

    def update(self, data):
        with self._lock:
            self._cycle = data.get("cycle", self._cycle)
            self._elapsed = data.get("elapsed", self._elapsed)
            self._mode = data.get("mode", self._mode)
            self._model_status = data.get("model_status", self._model_status)
            self._reroute_active = data.get("reroute_active", self._reroute_active)
            self._reroute_path = data.get("reroute_path", self._reroute_path)
            self._train_time_ms = data.get("train_time_ms", self._train_time_ms)
            self._predict_time_ms = data.get("predict_time_ms", self._predict_time_ms)
            if data.get("r2_score") is not None:
                self._r2_score = data["r2_score"]
            if data.get("mae_score") is not None:
                self._mae_score = data["mae_score"]

            preds = data.get("predictions", {})
            self._predictions = preds
            self._congested = data.get("congested", [])

            if preds:
                self._times.append(round(self._elapsed, 1))
                for key, val in preds.items():
                    self._util_history[key].append(round(val, 4))

            self._nic_tx.append(round(data.get("nic_tx_mbps", 0), 3))
            self._nic_rx.append(round(data.get("nic_rx_mbps", 0), 3))

            if "default_path" in data:
                self._default_path = data["default_path"]
            if "active_path" in data:
                self._active_path = data["active_path"]
            if "device_info" in data and data["device_info"]:
                self._device_info = data["device_info"]

            event = data.get("event")
            if event:
                ts = time.strftime("%H:%M:%S")
                self._events.appendleft({"time": ts, "msg": event})

    def add_event(self, msg):
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            self._events.appendleft({"time": ts, "msg": msg})

    def snapshot(self):
        with self._lock:
            # Build link utilization from predictions
            link_utils = {}
            for key, val in self._predictions.items():
                link_utils[key] = round(val, 4)

            # Key links for chart
            key_links = [f"s1:{i}" for i in range(2, self._n_switches + 1)]
            if self._n_switches >= 3:
                key_links.append("s2:3")

            util_series = {}
            for key in key_links:
                util_series[key] = list(self._util_history.get(key, []))

            return {
                "n_switches": self._n_switches,
                "cycle": self._cycle,
                "elapsed": round(self._elapsed, 1),
                "mode": self._mode,
                "model_status": self._model_status,
                "reroute_active": self._reroute_active,
                "reroute_path": self._reroute_path,
                "train_time_ms": round(self._train_time_ms, 1),
                "predict_time_ms": round(self._predict_time_ms, 2),
                "r2_score": round(self._r2_score, 4) if self._r2_score is not None else None,
                "mae_score": round(self._mae_score, 6) if self._mae_score is not None else None,
                "times": list(self._times),
                "util_series": util_series,
                "nic_tx": list(self._nic_tx),
                "nic_rx": list(self._nic_rx),
                "predictions": link_utils,
                "congested": self._congested,
                "default_path": self._default_path,
                "active_path": self._active_path,
                "device_info": self._device_info,
                "events": list(self._events)[:25],
            }


_state = DashboardState()
_routing_engine = None  # Set by WebDashboard.start()


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/data":
            self._send_json(_state.snapshot())
        elif path == "/api/topology":
            if _routing_engine:
                self._send_json(_routing_engine.get_topology_snapshot())
            else:
                self._send_json({"switches": [], "hosts": [], "links": []})
        elif path == "/" or path == "/index.html":
            self._send_html(DASHBOARD_HTML)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if not _routing_engine:
            self._send_json({"ok": False, "error": "No routing engine"})
            return

        if path == "/api/topology/add_switch":
            sid = data.get("switch_id", "")
            hid = data.get("host_id")
            hip = data.get("host_ip")
            ok = _routing_engine.add_switch(sid, hid, hip)
            self._send_json({"ok": ok, "topology": _routing_engine.get_topology_snapshot()})

        elif path == "/api/topology/remove_switch":
            sid = data.get("switch_id", "")
            ok = _routing_engine.remove_switch(sid)
            self._send_json({"ok": ok, "topology": _routing_engine.get_topology_snapshot()})

        elif path == "/api/topology/add_link":
            sw1 = data.get("from", "")
            sw2 = data.get("to", "")
            weight = data.get("weight", 1.0)
            ok = _routing_engine.add_link(sw1, sw2, weight)
            self._send_json({"ok": ok, "topology": _routing_engine.get_topology_snapshot()})

        elif path == "/api/topology/remove_link":
            sw1 = data.get("from", "")
            sw2 = data.get("to", "")
            ok = _routing_engine.remove_link(sw1, sw2)
            self._send_json({"ok": ok, "topology": _routing_engine.get_topology_snapshot()})

        elif path == "/api/topology/find_path":
            src = data.get("from", "")
            dst = data.get("to", "")
            path_result = _routing_engine.dijkstra_shortest_path(src, dst)
            self._send_json({"ok": True, "path": path_result})

        elif path == "/api/topology/apply":
            # Apply a full custom topology from the editor
            switches = data.get("switches", [])
            links = data.get("links", [])
            hosts = data.get("hosts", [])

            # Rebuild the routing engine graph from editor data
            _routing_engine.graph = {}
            _routing_engine._link_weights = {}
            _routing_engine._switch_links = set()
            # Clear port mappings
            _routing_engine.PORT_TO_NEIGHBOR = {}

            for sw_id in switches:
                _routing_engine.graph[sw_id] = set()

            for link in links:
                sw1, sw2 = link["from"], link["to"]
                if sw1 in _routing_engine.graph and sw2 in _routing_engine.graph:
                    _routing_engine.graph[sw1].add(sw2)
                    _routing_engine.graph[sw2].add(sw1)
                    key = frozenset({sw1, sw2})
                    _routing_engine._link_weights[key] = link.get("weight", 1.0)
                    _routing_engine._switch_links.add((sw1, sw2))

            # Rebuild port mappings
            for sw_id in switches:
                port = 2
                for nbr in sorted(_routing_engine.graph.get(sw_id, set())):
                    _routing_engine.PORT_TO_NEIGHBOR[(sw_id, port)] = nbr
                    port += 1

            # Rebuild host mappings
            _routing_engine._host_switch = {}
            _routing_engine._ip_host = {}
            for h in hosts:
                hid = h["id"]
                sw = h["switch"]
                ip = h.get("ip", "")
                _routing_engine._host_switch[hid] = sw
                if ip:
                    _routing_engine._ip_host[ip] = hid
                _routing_engine.PORT_TO_NEIGHBOR[(sw, 1)] = hid

            # Update n_switches in dashboard
            _state._n_switches = len(switches)

            _state.add_event(f"✏️ Topology edited: {len(switches)} switches, {len(links)} links")
            print(f"[Dashboard] Topology applied from editor: {_routing_engine.graph}")

            self._send_json({"ok": True, "topology": _routing_engine.get_topology_snapshot()})
        else:
            self.send_error(404)

    def _send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class WebDashboard:
    def __init__(self, port=8050, n_switches=4, routing_engine=None):
        self.port = port
        self.n_switches = n_switches
        self.routing_engine = routing_engine
        self._server = None
        self._thread = None

    def start(self, open_browser=True):
        global _state, _routing_engine
        _state = DashboardState(n_switches=self.n_switches)
        _routing_engine = self.routing_engine

        self._server = HTTPServer(("0.0.0.0", self.port), DashboardHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://localhost:{self.port}"
        print(f"  Dashboard running at: {url}")
        if open_browser:
            webbrowser.open(url)

    def update(self, data):
        _state.update(data)

    def add_event(self, msg):
        _state.add_event(msg)

    def stop(self):
        if self._server:
            self._server.shutdown()


# =====================================================================
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SDN Congestion Avoidance — Live Dashboard &amp; Editor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0e17;--bg2:#0f1420;--panel:#141925;
  --border:rgba(99,140,255,0.08);--glow:rgba(88,166,255,0.04);
  --text:#e2e8f0;--dim:#64748b;--muted:#475569;
  --green:#22c55e;--yellow:#eab308;--red:#ef4444;
  --blue:#3b82f6;--cyan:#06b6d4;--purple:#a855f7;--orange:#f97316;
  --accent:#58a6ff;--radius:12px;
}
body{
  background:var(--bg);color:var(--text);
  font-family:'Inter',-apple-system,sans-serif;line-height:1.5;
  min-height:100vh;overflow-x:hidden;
}
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 80% 60% at 20% 10%,rgba(88,166,255,0.06) 0%,transparent 60%),
    radial-gradient(ellipse 60% 50% at 80% 80%,rgba(168,85,247,0.05) 0%,transparent 60%);
}
.container{max-width:1480px;margin:0 auto;padding:16px 20px;position:relative;z-index:1}

/* Header */
.header{text-align:center;padding:14px 0 10px;margin-bottom:14px;border-bottom:1px solid var(--border)}
.header h1{
  font-size:20px;font-weight:800;letter-spacing:2px;
  background:linear-gradient(135deg,var(--accent),var(--cyan),var(--purple));
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.header-sub{
  margin-top:4px;font-size:12px;color:var(--dim);
  display:flex;align-items:center;justify-content:center;gap:14px;
}
.live-dot{
  width:8px;height:8px;border-radius:50%;display:inline-block;
  background:var(--green);box-shadow:0 0 8px var(--green);
  animation:pulse 2s ease-in-out infinite;
}
.live-dot.alert{background:var(--red);box-shadow:0 0 8px var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* Stat cards */
.stats-row{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:14px}
.stat-card{
  background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);
  padding:10px 14px;transition:border-color .3s,box-shadow .3s;
}
.stat-card:hover{border-color:rgba(88,166,255,.2);box-shadow:0 0 20px var(--glow)}
.stat-card .label{font-size:9px;font-weight:600;color:var(--dim);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:4px}
.stat-card .value{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace}
.stat-card .unit{font-size:10px;color:var(--dim);font-weight:400;margin-left:2px}
.stat-card.green .value{color:var(--green)}.stat-card.yellow .value{color:var(--yellow)}
.stat-card.red .value{color:var(--red)}.stat-card.blue .value{color:var(--accent)}
.stat-card.cyan .value{color:var(--cyan)}.stat-card.purple .value{color:var(--purple)}

/* Main grid */
.grid-main{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.panel{
  background:var(--panel);border:1px solid var(--border);
  border-radius:var(--radius);padding:14px 16px;position:relative;
}
.panel-title{
  font-size:10px;font-weight:700;color:var(--dim);text-transform:uppercase;
  letter-spacing:1.5px;margin-bottom:10px;display:flex;align-items:center;gap:6px;
}
.panel-title .dot{width:6px;height:6px;border-radius:50%;display:inline-block}
.chart-wrap{position:relative;height:200px}

/* Topology panel */
.topo-panel{grid-column:1/-1}
.topo-canvas-wrap{position:relative;height:340px;border-radius:8px;overflow:hidden;background:var(--bg2);border:1px solid var(--border)}
#topoCanvas{width:100%;height:100%}

/* Gauges */
.gauges-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.gauge-item{background:var(--bg2);border-radius:8px;padding:10px 12px;border:1px solid var(--border)}
.gauge-label{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.gauge-label span:first-child{font-size:11px;font-weight:600;color:var(--text)}
.gauge-label span:last-child{font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace}
.gauge-bar{height:6px;border-radius:3px;background:rgba(255,255,255,.06);overflow:hidden}
.gauge-fill{height:100%;border-radius:3px;transition:width .8s cubic-bezier(.22,1,.36,1),background .5s}

/* Events */
.event-log{grid-column:1/-1}
.events-list{max-height:180px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.events-list::-webkit-scrollbar{width:4px}
.events-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.event-item{
  display:flex;gap:8px;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.03);
  font-size:11px;font-family:'JetBrains Mono',monospace;animation:fadeIn .3s ease;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1}}
.event-time{color:var(--muted);flex-shrink:0;min-width:60px}
.event-msg{color:var(--dim)}
.event-msg.congestion{color:var(--red);font-weight:600}
.event-msg.reroute{color:var(--yellow)}.event-msg.cleared{color:var(--green)}
.event-msg.train{color:var(--cyan)}.event-msg.info{color:var(--dim)}

/* Legend for topology */
.topo-legend{
  position:absolute;bottom:8px;left:12px;display:flex;gap:14px;
  font-size:9px;color:var(--dim);z-index:2;
}
.topo-legend span{display:flex;align-items:center;gap:4px}
.topo-legend .swatch{width:16px;height:3px;border-radius:2px;display:inline-block}

@media(max-width:1100px){.stats-row{grid-template-columns:repeat(3,1fr)}.grid-main{grid-template-columns:1fr}}
@media(max-width:600px){.stats-row{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>SDN PROACTIVE CONGESTION AVOIDANCE</h1>
    <div class="header-sub">
      <span><span class="live-dot" id="liveDot"></span>&nbsp;<span id="statusLabel">LIVE</span></span>
      <span id="headerCycle">Cycle 0</span>
      <span id="headerElapsed">0s</span>
      <span id="headerMode">SIMULATE</span>
      <span id="headerSwitches">4 Switches</span>
    </div>
  </div>

  <div class="stats-row">
    <div class="stat-card cyan"><div class="label">Cycle</div><div class="value" id="valCycle">0</div></div>
    <div class="stat-card blue" id="cardModel"><div class="label">Model</div><div class="value" id="valModel" style="font-size:13px">collecting</div></div>
    <div class="stat-card green" id="cardR2"><div class="label">R² Score</div><div class="value" id="valR2">—</div></div>
    <div class="stat-card purple"><div class="label">Predict</div><div class="value" id="valPredict">—<span class="unit">ms</span></div></div>
    <div class="stat-card cyan"><div class="label">Train</div><div class="value" id="valTrain">—<span class="unit">ms</span></div></div>
    <div class="stat-card green" id="cardReroute"><div class="label">Reroute</div><div class="value" id="valReroute">Idle</div></div>
  </div>

  <div class="grid-main">
    <!-- TOPOLOGY -->
    <div class="panel topo-panel">
      <div class="panel-title"><span class="dot" style="background:var(--accent)"></span> Network Topology — Live Routing</div>
      <div class="topo-canvas-wrap">
        <canvas id="topoCanvas"></canvas>
        <div class="topo-legend">
          <span><span class="swatch" style="background:var(--green)"></span> Healthy</span>
          <span><span class="swatch" style="background:var(--yellow)"></span> Moderate</span>
          <span><span class="swatch" style="background:var(--red)"></span> Congested</span>
          <span><span class="swatch" style="background:var(--blue);height:4px"></span> Active Path</span>
          <span><span class="swatch" style="background:var(--orange);height:4px"></span> Reroute Path</span>
        </div>
      </div>
    </div>

    <!-- UTIL CHART -->
    <div class="panel">
      <div class="panel-title"><span class="dot" style="background:var(--blue)"></span> Link Utilization (Predicted)</div>
      <div class="chart-wrap"><canvas id="chartUtil"></canvas></div>
    </div>

    <!-- GAUGES + ML -->
    <div class="panel">
      <div class="panel-title"><span class="dot" style="background:var(--green)"></span> Link Gauges &amp; ML Metrics</div>
      <div id="gaugesContainer" class="gauges-grid"></div>
    </div>

    <!-- NIC CHART -->
    <div class="panel">
      <div class="panel-title"><span class="dot" style="background:var(--cyan)"></span> Network Throughput</div>
      <div class="chart-wrap"><canvas id="chartNic"></canvas></div>
    </div>

    <!-- ML -->
    <div class="panel">
      <div class="panel-title"><span class="dot" style="background:var(--purple)"></span> ML Performance</div>
      <div class="gauges-grid">
        <div class="gauge-item"><div class="gauge-label"><span>R²</span><span id="mlR2" style="color:var(--green)">—</span></div><div class="gauge-bar"><div class="gauge-fill" id="mlR2bar" style="width:0%;background:var(--green)"></div></div></div>
        <div class="gauge-item"><div class="gauge-label"><span>MAE</span><span id="mlMAE" style="color:var(--cyan)">—</span></div><div class="gauge-bar"><div class="gauge-fill" id="mlMAEbar" style="width:0%;background:var(--cyan)"></div></div></div>
        <div class="gauge-item"><div class="gauge-label"><span>Train</span><span id="mlTrain" style="color:var(--purple)">—</span></div><div class="gauge-bar"><div class="gauge-fill" id="mlTrainBar" style="width:0%;background:var(--purple)"></div></div></div>
        <div class="gauge-item"><div class="gauge-label"><span>Predict</span><span id="mlPred" style="color:var(--accent)">—</span></div><div class="gauge-bar"><div class="gauge-fill" id="mlPredBar" style="width:0%;background:var(--accent)"></div></div></div>
      </div>
    </div>

    <!-- EVENTS -->
    <div class="panel event-log">
      <div class="panel-title"><span class="dot" style="background:var(--yellow)"></span> Event Log</div>
      <div class="events-list" id="eventsList">
        <div class="event-item"><span class="event-time">--:--:--</span><span class="event-msg info">Waiting for data...</span></div>
      </div>
    </div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════
//  CONSTANTS
// ═══════════════════════════════════════════════════════
const THRESHOLD = 0.8;
const LINK_COLORS = ['#3b82f6','#22c55e','#a855f7','#f97316','#06b6d4','#ec4899','#eab308','#14b8a6','#f43f5e','#8b5cf6'];

Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
Chart.defaults.font.family = "'Inter', sans-serif";

// ═══════════════════════════════════════════════════════
//  TOPOLOGY VISUALIZATION (Canvas)
// ═══════════════════════════════════════════════════════
class TopologyViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.n = 4;
    this.switches = [];
    this.hosts = [];
    this.linkUtils = {};
    this.congestedLinks = [];
    this.defaultPath = [];
    this.activePath = [];
    this.rerouteActive = false;
    this.packetPhase = 0;
    this.time = 0;
    this.dpr = window.devicePixelRatio || 1;

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.animate();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = rect.height + 'px';
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.w = rect.width;
    this.h = rect.height;
    this.computeLayout();
  }

  computeLayout() {
    const cx = this.w / 2;
    const cy = this.h / 2;
    const sr = Math.min(this.w, this.h) * 0.26;
    const hr = Math.min(this.w, this.h) * 0.40;

    this.switches = [];
    this.hosts = [];
    for (let i = 0; i < this.n; i++) {
      const a = (2 * Math.PI * i / this.n) - Math.PI / 2;
      const di_sw = this.deviceInfo?.switches?.[i];
      const di_h = this.deviceInfo?.hosts?.[i];
      this.switches.push({
        id: 's'+(i+1),
        x: cx + sr*Math.cos(a), y: cy + sr*Math.sin(a),
        realIp: di_sw?.ip || '',
        realName: di_sw?.hostname || di_sw?.role || '',
        role: di_sw?.role || 'switch',
      });
      this.hosts.push({
        id: 'h'+(i+1),
        x: cx + hr*Math.cos(a), y: cy + hr*Math.sin(a),
        ip: di_h?.ip || '10.0.0.'+(i+1),
        realName: di_h?.hostname || '',
      });
    }
  }

  setData(d) {
    if (d.device_info) this.deviceInfo = d.device_info;
    if (d.n_switches !== this.n) { this.n = d.n_switches; this.computeLayout(); }
    this.linkUtils = d.predictions || {};
    this.congestedLinks = (d.congested || []).map(c => c[0]+':'+c[1]);
    this.defaultPath = d.default_path || [];
    this.activePath = d.active_path || [];
    this.rerouteActive = d.reroute_active || false;
  }

  getSwitchPos(id) { return this.switches.find(s => s.id === id); }

  // Find utilization for the link between two switches
  getLinkUtil(s1, s2) {
    // port_to_neighbor: s1 port 2 → s2, etc.
    // We need to find which port on s1 connects to s2
    const i1 = parseInt(s1.slice(1)) - 1;
    const i2 = parseInt(s2.slice(1)) - 1;
    // Port assignment: port 1 = host, then other switches in order
    // For s(i1+1), the switches in order are s1,s2,...,sN excluding self
    // Port for s2 on s1: skip self, count position
    let port = 2;
    for (let j = 0; j < this.n; j++) {
      if (j === i1) continue;
      if (j === i2) break;
      port++;
    }
    const key = s1 + ':' + port;
    return this.linkUtils[key] || 0;
  }

  isLinkCongested(s1, s2) {
    const i1 = parseInt(s1.slice(1)) - 1;
    const i2 = parseInt(s2.slice(1)) - 1;
    let port = 2;
    for (let j = 0; j < this.n; j++) {
      if (j === i1) continue;
      if (j === i2) break;
      port++;
    }
    return this.congestedLinks.includes(s1+':'+port);
  }

  isOnPath(path, s1, s2) {
    for (let i = 0; i < path.length - 1; i++) {
      if ((path[i] === s1 && path[i+1] === s2) || (path[i] === s2 && path[i+1] === s1)) return true;
    }
    return false;
  }

  utilColor(u) {
    if (u >= 0.8) return '#ef4444';
    if (u >= 0.5) return '#eab308';
    if (u >= 0.2) return '#22c55e';
    return 'rgba(100,116,139,0.3)';
  }

  animate() {
    this.time += 0.016;
    this.packetPhase = (this.packetPhase + 0.008) % 1;
    this.draw();
    requestAnimationFrame(() => this.animate());
  }

  draw() {
    const ctx = this.ctx;
    const w = this.w, h = this.h;
    ctx.clearRect(0, 0, w, h);

    // Dot grid background
    ctx.fillStyle = 'rgba(255,255,255,0.015)';
    for (let x = 15; x < w; x += 25) for (let y = 15; y < h; y += 25) {
      ctx.beginPath(); ctx.arc(x, y, 0.8, 0, Math.PI*2); ctx.fill();
    }

    // Draw all switch-switch links
    for (let i = 0; i < this.n; i++) {
      for (let j = i+1; j < this.n; j++) {
        const s1 = this.switches[i], s2 = this.switches[j];
        const util = this.getLinkUtil(s1.id, s2.id);
        const congested = this.isLinkCongested(s1.id, s2.id);
        const onDefault = this.isOnPath(this.defaultPath, s1.id, s2.id);
        const onActive = this.isOnPath(this.activePath, s1.id, s2.id);

        ctx.beginPath();
        ctx.moveTo(s1.x, s1.y);
        ctx.lineTo(s2.x, s2.y);

        if (congested) {
          // Pulsing red
          const glow = 0.6 + 0.4 * Math.sin(this.time * 6);
          ctx.strokeStyle = `rgba(239,68,68,${glow})`;
          ctx.lineWidth = 3.5;
          ctx.setLineDash([]);
        } else if (onActive && this.rerouteActive) {
          ctx.strokeStyle = '#f97316';
          ctx.lineWidth = 3.5;
          ctx.setLineDash([10, 5]);
          ctx.lineDashOffset = -this.time * 40;
        } else if (onDefault && !this.rerouteActive) {
          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = 3;
          ctx.setLineDash([10, 5]);
          ctx.lineDashOffset = -this.time * 30;
        } else {
          ctx.strokeStyle = this.utilColor(util);
          ctx.lineWidth = 1.5 + util * 2;
          ctx.setLineDash([]);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Utilization label on link midpoint
        if (util > 0.01) {
          const mx = (s1.x + s2.x) / 2, my = (s1.y + s2.y) / 2;
          ctx.font = '500 9px JetBrains Mono';
          ctx.fillStyle = congested ? '#ef4444' : 'rgba(226,232,240,0.5)';
          ctx.textAlign = 'center';
          ctx.fillText(Math.round(util*100)+'%', mx, my - 6);
        }
      }
    }

    // Draw host-switch links
    for (let i = 0; i < this.n; i++) {
      ctx.beginPath();
      ctx.moveTo(this.hosts[i].x, this.hosts[i].y);
      ctx.lineTo(this.switches[i].x, this.switches[i].y);
      ctx.strokeStyle = 'rgba(100,116,139,0.25)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Animate packets along active path
    const path = this.rerouteActive && this.activePath.length > 1 ? this.activePath : this.defaultPath;
    if (path.length > 1) {
      // Draw multiple packets at different phases
      for (let pkt = 0; pkt < 3; pkt++) {
        const phase = (this.packetPhase + pkt * 0.33) % 1;
        const totalSegs = path.length - 1;
        const globalPos = phase * totalSegs;
        const segIndex = Math.floor(globalPos);
        const segPhase = globalPos - segIndex;

        if (segIndex < totalSegs) {
          const from = this.getSwitchPos(path[segIndex]);
          const to = this.getSwitchPos(path[segIndex + 1]);
          if (from && to) {
            const px = from.x + (to.x - from.x) * segPhase;
            const py = from.y + (to.y - from.y) * segPhase;

            const color = this.rerouteActive ? '#f97316' : '#3b82f6';

            // Glow
            const grad = ctx.createRadialGradient(px, py, 0, px, py, 12);
            grad.addColorStop(0, color);
            grad.addColorStop(1, 'transparent');
            ctx.fillStyle = grad;
            ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI*2); ctx.fill();

            // Dot
            ctx.fillStyle = '#fff';
            ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI*2); ctx.fill();
          }
        }
      }

      // Also animate from first host to first switch, and last switch to last host
      const firstSw = this.getSwitchPos(path[0]);
      const lastSw = this.getSwitchPos(path[path.length-1]);
      const srcHost = this.hosts[parseInt(path[0].slice(1))-1];
      const dstHost = this.hosts[parseInt(path[path.length-1].slice(1))-1];
      if (firstSw && srcHost) {
        const p = (this.packetPhase * 2) % 1;
        const px = srcHost.x + (firstSw.x - srcHost.x) * p;
        const py = srcHost.y + (firstSw.y - srcHost.y) * p;
        ctx.fillStyle = 'rgba(88,166,255,0.3)';
        ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI*2); ctx.fill();
      }
      if (lastSw && dstHost) {
        const p = (this.packetPhase * 2) % 1;
        const px = lastSw.x + (dstHost.x - lastSw.x) * p;
        const py = lastSw.y + (dstHost.y - lastSw.y) * p;
        ctx.fillStyle = 'rgba(88,166,255,0.3)';
        ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI*2); ctx.fill();
      }
    }

    // Draw switches
    for (const sw of this.switches) {
      const congested = this.congestedLinks.some(c => c.startsWith(sw.id+':'));
      const swW = 52, swH = 28, r = 8;

      // Glow
      if (congested) {
        const glow = 0.15 + 0.1 * Math.sin(this.time * 6);
        ctx.shadowColor = '#ef4444';
        ctx.shadowBlur = 15 * glow * 5;
      } else if (this.activePath.includes(sw.id) || this.defaultPath.includes(sw.id)) {
        ctx.shadowColor = this.rerouteActive ? '#f97316' : '#3b82f6';
        ctx.shadowBlur = 10;
      }

      // Box
      ctx.beginPath();
      ctx.roundRect(sw.x - swW/2, sw.y - swH/2, swW, swH, r);
      ctx.fillStyle = congested ? 'rgba(239,68,68,0.15)' : 'rgba(20,25,37,0.95)';
      ctx.fill();
      ctx.strokeStyle = congested ? '#ef4444' : 'rgba(88,166,255,0.4)';
      ctx.lineWidth = congested ? 2 : 1.5;
      ctx.stroke();

      ctx.shadowBlur = 0;

      // Label
      ctx.font = '600 11px Inter';
      ctx.fillStyle = congested ? '#ef4444' : var_text;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(sw.id.toUpperCase(), sw.x, sw.y - (sw.realIp ? 4 : 0));

      // Real IP below name
      if (sw.realIp) {
        ctx.font = '400 8px JetBrains Mono';
        ctx.fillStyle = 'rgba(88,166,255,0.6)';
        ctx.fillText(sw.realIp, sw.x, sw.y + 8);
      }

      // Role badge above switch
      if (sw.role === 'gateway') {
        ctx.font = '700 7px Inter';
        ctx.fillStyle = '#06b6d4';
        ctx.fillText('GATEWAY', sw.x, sw.y - swH/2 - 6);
      } else if (sw.role === 'isp_router') {
        ctx.font = '700 7px Inter';
        ctx.fillStyle = '#a855f7';
        ctx.fillText('ISP', sw.x, sw.y - swH/2 - 6);
      }
    }

    // Draw hosts
    for (const host of this.hosts) {
      const radius = 16;

      ctx.beginPath();
      ctx.arc(host.x, host.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(20,25,37,0.9)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(34,197,94,0.5)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.font = '600 10px Inter';
      ctx.fillStyle = '#22c55e';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(host.id.toUpperCase(), host.x, host.y - (host.realName ? 4 : 0));

      // Real hostname above
      if (host.realName) {
        ctx.font = '400 7px JetBrains Mono';
        ctx.fillStyle = 'rgba(34,197,94,0.5)';
        const name = host.realName.length > 12 ? host.realName.slice(0,12)+'…' : host.realName;
        ctx.fillText(name, host.x, host.y - radius - 6);
      }

      // IP below
      ctx.font = '400 8px JetBrains Mono';
      ctx.fillStyle = 'rgba(100,116,139,0.6)';
      ctx.fillText(host.ip, host.x, host.y + radius + 10);
    }

    // Path label
    if (path.length > 1) {
      const label = this.rerouteActive ? 'REROUTE: ' : 'PATH: ';
      const pathStr = path.join(' → ');
      ctx.font = '600 10px JetBrains Mono';
      ctx.fillStyle = this.rerouteActive ? '#f97316' : '#3b82f6';
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(label + pathStr, 12, 10);
    }
  }
}

const var_text = '#e2e8f0';

// ═══════════════════════════════════════════════════════
//  CHARTS
// ═══════════════════════════════════════════════════════
const ctxUtil = document.getElementById('chartUtil').getContext('2d');
const chartUtil = new Chart(ctxUtil, {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: {
    responsive:true, maintainAspectRatio:false,
    animation:{duration:500,easing:'easeInOutQuart'},
    scales:{
      x:{display:true,title:{display:true,text:'Time (s)',font:{size:9}},ticks:{maxTicksLimit:8,font:{size:8}}},
      y:{display:true,min:0,max:1.05,title:{display:true,text:'Utilization',font:{size:9}},ticks:{callback:v=>(v*100)+'%',font:{size:8}}}
    },
    plugins:{legend:{position:'top',labels:{boxWidth:10,padding:10,font:{size:9}}}},
    interaction:{mode:'index',intersect:false},
  }
});

const ctxNic = document.getElementById('chartNic').getContext('2d');
const chartNic = new Chart(ctxNic, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {label:'TX',data:[],borderColor:'#06b6d4',backgroundColor:'rgba(6,182,212,0.12)',borderWidth:2,tension:.35,fill:true,pointRadius:0},
      {label:'RX',data:[],borderColor:'#a855f7',backgroundColor:'rgba(168,85,247,0.10)',borderWidth:2,tension:.35,fill:true,pointRadius:0},
    ]
  },
  options: {
    responsive:true,maintainAspectRatio:false,
    animation:{duration:500,easing:'easeInOutQuart'},
    scales:{
      x:{display:true,title:{display:true,text:'Time (s)',font:{size:9}},ticks:{maxTicksLimit:8,font:{size:8}}},
      y:{display:true,min:0,title:{display:true,text:'Mbps',font:{size:9}},ticks:{font:{size:8}}}
    },
    plugins:{legend:{position:'top',labels:{boxWidth:10,padding:10,font:{size:9}}}},
    interaction:{mode:'index',intersect:false},
  }
});

// ═══════════════════════════════════════════════════════
//  TOPOLOGY
// ═══════════════════════════════════════════════════════
const topoViz = new TopologyViz(document.getElementById('topoCanvas'));

// ═══════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════
function gaugeColor(v){if(v>=.8)return'var(--red)';if(v>=.5)return'var(--yellow)';return'var(--green)'}
function eventClass(m){
  const u=m.toUpperCase();
  if(u.includes('CONGESTION')||u.includes('🚨'))return'congestion';
  if(u.includes('REROUTE')||u.includes('⤷')||u.includes('FLOWMOD'))return'reroute';
  if(u.includes('CLEARED')||u.includes('✓')||u.includes('✅'))return'cleared';
  if(u.includes('TRAIN')||u.includes('MODEL')||u.includes('⏳'))return'train';
  return'info';
}

let utilDatasetsBuilt = false;
let gaugesBuilt = false;
let prevEventCount = 0;

// ═══════════════════════════════════════════════════════
//  POLL
// ═══════════════════════════════════════════════════════
async function poll() {
  try {
    const res = await fetch('/api/data');
    const d = await res.json();

    // Header
    document.getElementById('headerCycle').textContent = 'Cycle ' + d.cycle;
    const m = Math.floor(d.elapsed/60), s = Math.floor(d.elapsed%60);
    document.getElementById('headerElapsed').textContent = m+'m '+s+'s';
    document.getElementById('headerMode').textContent = d.mode.toUpperCase();
    document.getElementById('headerSwitches').textContent = d.n_switches + ' Switches';

    const dot = document.getElementById('liveDot');
    const label = document.getElementById('statusLabel');
    if (d.reroute_active) { dot.className='live-dot alert'; label.textContent='REROUTING'; label.style.color='var(--red)'; }
    else { dot.className='live-dot'; label.textContent='LIVE'; label.style.color=''; }

    // Stats
    document.getElementById('valCycle').textContent = d.cycle;
    document.getElementById('valModel').textContent = d.model_status;
    const cm = document.getElementById('cardModel');
    cm.className = d.model_status.includes('trained') ? 'stat-card green' : d.model_status.includes('training') ? 'stat-card yellow' : 'stat-card blue';
    document.getElementById('valR2').textContent = d.r2_score !== null ? d.r2_score.toFixed(4) : '—';
    const cr2 = document.getElementById('cardR2');
    if (d.r2_score !== null) cr2.className = d.r2_score > 0.7 ? 'stat-card green' : 'stat-card yellow';
    document.getElementById('valPredict').innerHTML = d.predict_time_ms > 0 ? d.predict_time_ms.toFixed(1)+'<span class="unit">ms</span>' : '—';
    document.getElementById('valTrain').innerHTML = d.train_time_ms > 0 ? Math.round(d.train_time_ms)+'<span class="unit">ms</span>' : '—';
    if (d.reroute_active) { document.getElementById('valReroute').textContent='ACTIVE'; document.getElementById('cardReroute').className='stat-card red'; }
    else { document.getElementById('valReroute').textContent='Idle'; document.getElementById('cardReroute').className='stat-card green'; }

    // Topology
    topoViz.setData(d);

    // Build util chart datasets dynamically
    const seriesKeys = Object.keys(d.util_series);
    if (seriesKeys.length > 0 && !utilDatasetsBuilt) {
      chartUtil.data.datasets = seriesKeys.map((key, i) => {
        // key is like "s1:2" → label "s1→s2" etc
        const parts = key.split(':');
        const sw = parts[0], port = parseInt(parts[1]);
        // Port 2+ maps to other switches
        const swIdx = parseInt(sw.slice(1)) - 1;
        let targetIdx = 0, p = 2;
        for (let j = 0; j < d.n_switches; j++) {
          if (j === swIdx) continue;
          if (p === port) { targetIdx = j; break; }
          p++;
        }
        const lbl = sw + '→s' + (targetIdx+1);
        const color = LINK_COLORS[i % LINK_COLORS.length];
        return { label: lbl, data: [], borderColor: color, backgroundColor: color+'18', borderWidth: 1.8, tension: .35, fill: false, pointRadius: 0 };
      });
      utilDatasetsBuilt = true;
    }

    if (utilDatasetsBuilt) {
      chartUtil.data.labels = d.times;
      seriesKeys.forEach((key, i) => {
        if (chartUtil.data.datasets[i]) chartUtil.data.datasets[i].data = d.util_series[key];
      });
      chartUtil.update('none');
    }

    // NIC chart
    if (d.nic_tx.length > 0) {
      const labels = d.times.length >= d.nic_tx.length ? d.times.slice(-d.nic_tx.length) : d.nic_tx.map((_,i)=>i);
      chartNic.data.labels = labels;
      chartNic.data.datasets[0].data = d.nic_tx;
      chartNic.data.datasets[1].data = d.nic_rx;
      chartNic.update('none');
    }

    // Gauges — build dynamically for N switches
    if (!gaugesBuilt && d.n_switches > 0) {
      const container = document.getElementById('gaugesContainer');
      container.innerHTML = '';
      // Key links: s1:2→sX, s1:3→sY, ..., s2:3→sZ
      const gaugeLinks = [];
      for (let p = 2; p <= d.n_switches; p++) gaugeLinks.push({sw:'s1',port:p});
      if (d.n_switches >= 3) gaugeLinks.push({sw:'s2',port:3});

      gaugeLinks.forEach(gl => {
        const swIdx = parseInt(gl.sw.slice(1)) - 1;
        let targetIdx = 0, pp = 2;
        for (let j = 0; j < d.n_switches; j++) {
          if (j === swIdx) continue;
          if (pp === gl.port) { targetIdx = j; break; }
          pp++;
        }
        const lbl = gl.sw + ' → s' + (targetIdx+1);
        const id = gl.sw.replace('s','') + 'p' + gl.port;
        const div = document.createElement('div');
        div.className = 'gauge-item';
        div.innerHTML = '<div class="gauge-label"><span>'+lbl+'</span><span id="gv-'+id+'">0%</span></div><div class="gauge-bar"><div class="gauge-fill" id="gf-'+id+'" style="width:0%;background:var(--green)"></div></div>';
        container.appendChild(div);
      });
      gaugesBuilt = true;
    }

    // Update gauges
    if (gaugesBuilt) {
      const gaugeLinks2 = [];
      for (let p = 2; p <= d.n_switches; p++) gaugeLinks2.push({sw:'s1',port:p});
      if (d.n_switches >= 3) gaugeLinks2.push({sw:'s2',port:3});

      gaugeLinks2.forEach(gl => {
        const key = gl.sw + ':' + gl.port;
        const val = d.predictions[key] || 0;
        const id = gl.sw.replace('s','') + 'p' + gl.port;
        const gv = document.getElementById('gv-'+id);
        const gf = document.getElementById('gf-'+id);
        if (gv) { gv.textContent = Math.round(val*100)+'%'; gv.style.color = gaugeColor(val); }
        if (gf) { gf.style.width = Math.min(100,Math.round(val*100))+'%'; gf.style.background = gaugeColor(val); }
      });
    }

    // ML
    if(d.r2_score!==null){document.getElementById('mlR2').textContent=d.r2_score.toFixed(4);document.getElementById('mlR2bar').style.width=Math.round(Math.max(0,d.r2_score)*100)+'%'}
    if(d.mae_score!==null){document.getElementById('mlMAE').textContent=d.mae_score.toFixed(6);document.getElementById('mlMAEbar').style.width=Math.min(100,Math.round((1-d.mae_score)*100))+'%'}
    if(d.train_time_ms>0){document.getElementById('mlTrain').textContent=Math.round(d.train_time_ms)+' ms';document.getElementById('mlTrainBar').style.width=Math.min(100,Math.round(d.train_time_ms/20))+'%'}
    if(d.predict_time_ms>0){document.getElementById('mlPred').textContent=d.predict_time_ms.toFixed(1)+' ms';document.getElementById('mlPredBar').style.width=Math.min(100,Math.round(d.predict_time_ms*20))+'%'}

    // Events
    if (d.events.length > 0 && d.events.length !== prevEventCount) {
      prevEventCount = d.events.length;
      const list = document.getElementById('eventsList');
      list.innerHTML = '';
      d.events.forEach(ev => {
        const div = document.createElement('div');
        div.className = 'event-item';
        div.innerHTML = '<span class="event-time">'+ev.time+'</span><span class="event-msg '+eventClass(ev.msg)+'">'+ev.msg+'</span>';
        list.appendChild(div);
      });
    }

  } catch(e) {}
}

setInterval(poll, 1000);
poll();
</script>
</body>
</html>"""
