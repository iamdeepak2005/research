#!/usr/bin/env python3
"""
utils.py — Shared constants and helper functions for the SDN prototype.
"""

import os
import json
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLL_INTERVAL = 5            # seconds between stats requests
PREDICT_HORIZON = 10         # predict utilisation this many seconds ahead
UTILIZATION_THRESHOLD = 0.80 # 80 % — trigger reroute above this
LINK_CAPACITY_BITS = 10_000_000  # 10 Mbps per link
LINK_CAPACITY_BYTES = LINK_CAPACITY_BITS // 8  # 1,250,000 bytes/s

TELEMETRY_CSV = os.path.join(os.path.dirname(__file__), "telemetry_data.csv")
TELEMETRY_JSON = os.path.join(os.path.dirname(__file__), "telemetry_latest.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# Switch adjacency — mirrors topology.py full mesh
# key = (sw1, sw2), value = True
SWITCH_LINKS = {
    ("s1", "s2"), ("s1", "s3"), ("s1", "s4"),
    ("s2", "s3"), ("s2", "s4"),
    ("s3", "s4"),
}

# Host-to-switch mapping
HOST_SWITCH = {
    "h1": "s1",
    "h2": "s2",
    "h3": "s3",
    "h4": "s4",
}

# IP-to-host mapping
IP_HOST = {
    "10.0.0.1": "h1",
    "10.0.0.2": "h2",
    "10.0.0.3": "h3",
    "10.0.0.4": "h4",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def now_ts():
    """Return current UTC timestamp as float."""
    return time.time()


def now_iso():
    """Return current time as ISO-8601 string."""
    return datetime.utcnow().isoformat()


def save_json(filepath, data):
    """Atomically write data to a JSON file."""
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, filepath)


def load_json(filepath):
    """Load JSON from file; return empty dict on failure."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def bytes_to_utilization(byte_count, duration_sec, capacity=LINK_CAPACITY_BYTES):
    """
    Convert a byte count over a duration into a 0-1 utilization ratio.
    """
    if duration_sec <= 0:
        return 0.0
    rate = byte_count / duration_sec  # bytes/sec
    return min(rate / capacity, 1.0)


# ---------------------------------------------------------------------------
# Dynamic topology generator
# ---------------------------------------------------------------------------
def generate_topology(n_switches=4):
    """
    Generate a full-mesh SDN topology with N switches and N hosts.

    Returns dict with:
        switches       — ["s1", "s2", ...]
        hosts          — ["h1", "h2", ...]
        switch_links   — {("s1","s2"), ("s1","s3"), ...}
        host_switch    — {"h1": "s1", ...}
        ip_host        — {"10.0.0.1": "h1", ...}
        port_to_neighbor — {("s1",1): "h1", ("s1",2): "s2", ...}
        switches_ports — [("s1",1), ("s1",2), ...]
    """
    switches = [f"s{i+1}" for i in range(n_switches)]
    hosts = [f"h{i+1}" for i in range(n_switches)]

    # Full mesh between all switches
    switch_links = set()
    for i in range(n_switches):
        for j in range(i + 1, n_switches):
            switch_links.add((switches[i], switches[j]))

    # Host ↔ switch mapping
    host_switch = {h: s for h, s in zip(hosts, switches)}

    # IP ↔ host mapping
    ip_host = {f"10.0.0.{i+1}": h for i, h in enumerate(hosts)}

    # Port-to-neighbor: port 1 = host, ports 2+ = other switches in order
    port_to_neighbor = {}
    for i, sw in enumerate(switches):
        port_to_neighbor[(sw, 1)] = hosts[i]
        port = 2
        for j, other in enumerate(switches):
            if i != j:
                port_to_neighbor[(sw, port)] = other
                port += 1

    # All (switch, port) pairs
    n_ports = n_switches  # 1 host port + (n-1) switch ports
    switches_ports = []
    for sw in switches:
        for p in range(1, n_ports + 1):
            switches_ports.append((sw, p))

    return {
        "switches": switches,
        "hosts": hosts,
        "switch_links": switch_links,
        "host_switch": host_switch,
        "ip_host": ip_host,
        "port_to_neighbor": port_to_neighbor,
        "switches_ports": switches_ports,
        "n_switches": n_switches,
    }
