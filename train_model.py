#!/usr/bin/env python3
"""
train_model.py — Train a Linear Regression model to predict link utilization
10 seconds ahead.

Usage:
    python3 train_model.py                   # use default telemetry_data.csv
    python3 train_model.py my_data.csv       # use custom CSV

Output:
    model.joblib   — serialised sklearn pipeline
    Prints R² score, MAE on the test split.

pip install:
    pip install scikit-learn pandas joblib
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

from utils import MODEL_PATH, TELEMETRY_CSV, PREDICT_HORIZON, POLL_INTERVAL


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure numeric
    for col in ["tx_bytes", "rx_bytes", "tx_packets", "rx_packets",
                 "duration_sec", "utilization"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df


def engineer_features(df):
    """
    For each (switch_id, port_no) group, compute rolling features and a
    future-utilization target shifted by PREDICT_HORIZON / POLL_INTERVAL rows.
    """
    shift_steps = max(1, PREDICT_HORIZON // POLL_INTERVAL)  # 2 rows at 5 s poll

    groups = []
    for (sw, port), g in df.groupby(["switch_id", "port_no"]):
        g = g.sort_values("timestamp").reset_index(drop=True)
        # byte / packet rates
        g["tx_rate"] = g["tx_bytes"].diff().fillna(0)
        g["rx_rate"] = g["rx_bytes"].diff().fillna(0)
        g["pkt_rate"] = (g["tx_packets"].diff().fillna(0) +
                         g["rx_packets"].diff().fillna(0))
        # rolling average utilisation (window=3)
        g["util_avg3"] = g["utilization"].rolling(3, min_periods=1).mean()
        g["util_avg5"] = g["utilization"].rolling(5, min_periods=1).mean()
        # target: utilisation shifted into the future
        g["target"] = g["utilization"].shift(-shift_steps)
        groups.append(g)

    out = pd.concat(groups, ignore_index=True)
    out.dropna(subset=["target"], inplace=True)
    return out


FEATURE_COLS = [
    "tx_bytes", "rx_bytes", "tx_packets", "rx_packets",
    "duration_sec", "utilization",
    "tx_rate", "rx_rate", "pkt_rate",
    "util_avg3", "util_avg5",
]


def train(csv_path=None):
    csv_path = csv_path or TELEMETRY_CSV
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        print("Run the controller + topology first to collect telemetry.")
        sys.exit(1)

    df = load_data(csv_path)
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")

    df = engineer_features(df)
    print(f"[INFO] After feature engineering: {len(df)} rows")

    if len(df) < 10:
        print("[ERROR] Not enough data to train. Collect more telemetry first.")
        sys.exit(1)

    X = df[FEATURE_COLS].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[RESULT] R² = {r2:.4f}")
    print(f"[RESULT] MAE = {mae:.6f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    train(csv_file)
