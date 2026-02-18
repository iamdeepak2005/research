#!/usr/bin/env python3
"""
telemetry_collector.py — In-memory ring-buffer for telemetry + JSON export.

Used by the Ryu controller to store per-port stats and make them available
to the predictor and routing engine.
"""

import collections
import threading
from utils import save_json, now_iso, TELEMETRY_JSON, bytes_to_utilization

MAX_SAMPLES = 100  # per (switch, port) pair


class TelemetryCollector:
    """Thread-safe telemetry store with ring-buffer per link."""

    def __init__(self):
        # key = (switch_id, port_no) → deque of records
        self._data = collections.defaultdict(
            lambda: collections.deque(maxlen=MAX_SAMPLES)
        )
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def add_record(self, record: dict):
        """
        Store one telemetry record.

        Expected keys:
            timestamp, switch_id, port_no,
            tx_packets, rx_packets, tx_bytes, rx_bytes,
            duration_sec
        """
        # Auto-compute utilization if not already present
        if "utilization" not in record:
            record["utilization"] = bytes_to_utilization(
                float(record.get("tx_bytes", 0)),
                float(record.get("duration_sec", 1)),
            )
        key = (record["switch_id"], record["port_no"])
        with self._lock:
            self._data[key].append(record)

    # ------------------------------------------------------------------
    def get_latest(self, switch_id, port_no):
        """Return the most recent record for a (switch, port), or None."""
        key = (switch_id, port_no)
        with self._lock:
            buf = self._data.get(key)
            if buf and len(buf) > 0:
                return buf[-1]
        return None

    # ------------------------------------------------------------------
    def get_window(self, switch_id, port_no, n=10):
        """Return the last *n* records for a (switch, port)."""
        key = (switch_id, port_no)
        with self._lock:
            buf = self._data.get(key)
            if buf:
                return list(buf)[-n:]
        return []

    # ------------------------------------------------------------------
    def all_latest(self):
        """Return dict mapping (switch, port) → latest record."""
        result = {}
        with self._lock:
            for key, buf in self._data.items():
                if buf:
                    result[key] = buf[-1]
        return result

    # ------------------------------------------------------------------
    def export_json(self, filepath=None):
        """Write the latest snapshot of all links to a JSON file."""
        filepath = filepath or TELEMETRY_JSON
        snapshot = {
            "timestamp": now_iso(),
            "links": [],
        }
        latest = self.all_latest()
        for (sw, port), rec in sorted(latest.items()):
            entry = dict(rec)
            entry["utilization"] = bytes_to_utilization(
                rec.get("tx_bytes", 0), rec.get("duration_sec", 1)
            )
            snapshot["links"].append(entry)
        save_json(filepath, snapshot)
        return snapshot
