#!/usr/bin/env python3
"""
dataset_logger.py — Appends telemetry records to a CSV file for ML training.
"""

import csv
import os
import threading
from utils import TELEMETRY_CSV, bytes_to_utilization

CSV_COLUMNS = [
    "timestamp",
    "switch_id",
    "port_no",
    "tx_packets",
    "rx_packets",
    "tx_bytes",
    "rx_bytes",
    "duration_sec",
    "utilization",
]


class DatasetLogger:
    """Append-only CSV writer for telemetry data."""

    def __init__(self, filepath=None):
        self.filepath = filepath or TELEMETRY_CSV
        self._lock = threading.Lock()
        self._ensure_header()

    # ------------------------------------------------------------------
    def _ensure_header(self):
        """Write the CSV header if the file does not exist yet."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    # ------------------------------------------------------------------
    def log(self, record: dict):
        """
        Append one telemetry record to the CSV.

        Computes 'utilization' automatically from tx_bytes and duration_sec.
        """
        row = {col: record.get(col, 0) for col in CSV_COLUMNS}
        row["utilization"] = bytes_to_utilization(
            float(record.get("tx_bytes", 0)),
            float(record.get("duration_sec", 1)),
        )
        with self._lock:
            with open(self.filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(row)

    # ------------------------------------------------------------------
    def log_many(self, records):
        """Append a batch of records."""
        for rec in records:
            self.log(rec)
