#!/usr/bin/env python3
"""
predictor.py — Real-time link utilization predictor.

Loads the trained model and predicts utilization 10 seconds ahead for each
(switch, port) pair.  Returns a list of links predicted to exceed the
congestion threshold.

pip install:
    pip install scikit-learn joblib numpy
"""

import os
import numpy as np
import joblib

from utils import MODEL_PATH, UTILIZATION_THRESHOLD


class Predictor:
    """Load a trained model and predict future utilization."""

    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"[Predictor] Model loaded from {self.model_path}")
        else:
            print(f"[Predictor] WARNING — model not found at {self.model_path}")
            print("[Predictor] Run train_model.py first. Using fallback (last value).")

    # ------------------------------------------------------------------
    def _feature_vector(self, window):
        """
        Build a single feature vector from the last few telemetry records.

        Expected record keys match train_model.FEATURE_COLS:
            tx_bytes, rx_bytes, tx_packets, rx_packets,
            duration_sec, utilization,
            tx_rate, rx_rate, pkt_rate,
            util_avg3, util_avg5
        """
        if not window:
            return None

        latest = window[-1]

        # Compute diff-based rates from window
        if len(window) >= 2:
            prev = window[-2]
            tx_rate = latest.get("tx_bytes", 0) - prev.get("tx_bytes", 0)
            rx_rate = latest.get("rx_bytes", 0) - prev.get("rx_bytes", 0)
            pkt_rate = (
                (latest.get("tx_packets", 0) - prev.get("tx_packets", 0))
                + (latest.get("rx_packets", 0) - prev.get("rx_packets", 0))
            )
        else:
            tx_rate = rx_rate = pkt_rate = 0.0

        utils = [r.get("utilization", 0) for r in window]
        util_avg3 = float(np.mean(utils[-3:])) if len(utils) >= 1 else 0.0
        util_avg5 = float(np.mean(utils[-5:])) if len(utils) >= 1 else 0.0

        vec = [
            latest.get("tx_bytes", 0),
            latest.get("rx_bytes", 0),
            latest.get("tx_packets", 0),
            latest.get("rx_packets", 0),
            latest.get("duration_sec", 1),
            latest.get("utilization", 0),
            tx_rate,
            rx_rate,
            pkt_rate,
            util_avg3,
            util_avg5,
        ]
        return np.array(vec, dtype=float).reshape(1, -1)

    # ------------------------------------------------------------------
    def predict(self, window):
        """
        Given a list of recent telemetry dicts for one (switch, port),
        return predicted utilization (float 0-1).
        """
        if not window:
            return 0.0

        vec = self._feature_vector(window)
        if vec is None:
            return 0.0

        if self.model is not None:
            pred = float(self.model.predict(vec)[0])
            return max(0.0, min(pred, 1.0))
        else:
            # Fallback: return latest utilization
            return window[-1].get("utilization", 0.0)

    # ------------------------------------------------------------------
    def predict_all(self, telemetry_collector):
        """
        Predict utilization for every (switch, port) in the collector.
        Returns:
            predictions  — dict {(switch, port): predicted_util}
            congested    — list of (switch, port) pairs above threshold
        """
        predictions = {}
        congested = []
        latest_all = telemetry_collector.all_latest()

        for (sw, port) in latest_all:
            window = telemetry_collector.get_window(sw, port, n=10)
            pred = self.predict(window)
            predictions[(sw, port)] = pred
            if pred >= UTILIZATION_THRESHOLD:
                congested.append((sw, port))

        return predictions, congested


# ------------------------------------------------------------------
# Quick standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    p = Predictor()
    # Dummy window for smoke test
    dummy = [
        {"tx_bytes": 1000 * i, "rx_bytes": 800 * i,
         "tx_packets": 10 * i, "rx_packets": 8 * i,
         "duration_sec": 5 * i if i > 0 else 1,
         "utilization": 0.1 * i}
        for i in range(1, 6)
    ]
    print(f"Predicted utilization: {p.predict(dummy):.4f}")
