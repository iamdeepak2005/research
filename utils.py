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
