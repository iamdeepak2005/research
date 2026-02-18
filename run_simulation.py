#!/usr/bin/env python3
"""
run_simulation.py — Full end-to-end simulation (runs on Windows, no Mininet needed).

Simulates the entire pipeline:
  1. Generates synthetic telemetry data (mimics what Ryu controller would collect)
  2. Logs it to CSV
  3. Trains the ML model
  4. Runs real-time predictions
  5. Triggers rerouting when congestion is detected

Usage:
    python run_simulation.py
"""

import time
import random
import math
import sys
import os

# Add project dir to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    LINK_CAPACITY_BYTES, UTILIZATION_THRESHOLD, TELEMETRY_CSV,
    MODEL_PATH, now_iso, bytes_to_utilization,
)
from telemetry_collector import TelemetryCollector
from dataset_logger import DatasetLogger
from predictor import Predictor
from routing_engine import RoutingEngine


# ======================================================================
#  PHASE 1: Generate synthetic telemetry data
# ======================================================================
def generate_synthetic_data(collector, logger, num_samples=200):
    """
    Simulate 200 rounds of port stats from 4 switches.
    
    Traffic pattern:
      - s1 port 2 (s1→s2 link): starts normal, ramps up to congestion
      - Other links: moderate random traffic
    """
    print("=" * 60)
    print("  PHASE 1: Generating Synthetic Telemetry Data")
    print("=" * 60)

    switches_ports = [
        ("s1", 1), ("s1", 2), ("s1", 3), ("s1", 4),
        ("s2", 1), ("s2", 2), ("s2", 3), ("s2", 4),
        ("s3", 1), ("s3", 2), ("s3", 3), ("s3", 4),
        ("s4", 1), ("s4", 2), ("s4", 3), ("s4", 4),
    ]

    for i in range(num_samples):
        ts = now_iso()
        interval = 5  # each sample represents a 5-second window

        for sw, port in switches_ports:
            # --- Traffic pattern (bytes in this 5-second interval) ---
            if sw == "s1" and port == 2:
                # Congestion ramp: starts at 20%, peaks at 95%
                progress = i / num_samples
                rate_fraction = 0.2 + 0.75 * progress
                noise = random.gauss(0, 0.03)
                rate_fraction = max(0.0, min(rate_fraction + noise, 1.0))
                tx_bytes = int(LINK_CAPACITY_BYTES * rate_fraction * interval)
            elif sw == "s2" and port == 3:
                # Moderate increasing traffic
                progress = i / num_samples
                rate_fraction = 0.1 + 0.5 * progress
                noise = random.gauss(0, 0.02)
                rate_fraction = max(0.0, min(rate_fraction + noise, 1.0))
                tx_bytes = int(LINK_CAPACITY_BYTES * rate_fraction * interval)
            else:
                # Low random traffic
                rate_fraction = random.uniform(0.05, 0.30)
                tx_bytes = int(LINK_CAPACITY_BYTES * rate_fraction * interval)

            rx_bytes = int(tx_bytes * random.uniform(0.6, 0.9))

            record = {
                "timestamp": ts,
                "switch_id": sw,
                "port_no": port,
                "tx_packets": max(1, tx_bytes // 1500),
                "rx_packets": max(1, rx_bytes // 1500),
                "tx_bytes": tx_bytes,
                "rx_bytes": rx_bytes,
                "duration_sec": interval,
            }

            collector.add_record(record)
            logger.log(record)

        if (i + 1) % 50 == 0:
            # Show progress with a sample utilization
            sample = collector.get_latest("s1", 2)
            util = bytes_to_utilization(sample["tx_bytes"], sample["duration_sec"])
            print(f"  Sample {i+1}/{num_samples} | s1:port2 utilization = {util:.1%}")

    print(f"\n  ✓ Generated {num_samples} samples × 16 ports = "
          f"{num_samples * 16} records")
    print(f"  ✓ Saved to {TELEMETRY_CSV}")


# ======================================================================
#  PHASE 2: Train the ML model
# ======================================================================
def train_model():
    print("\n" + "=" * 60)
    print("  PHASE 2: Training ML Model (Linear Regression)")
    print("=" * 60)

    from train_model import train
    train()


# ======================================================================
#  PHASE 3: Run predictions on the collected data
# ======================================================================
def run_predictions(collector):
    print("\n" + "=" * 60)
    print("  PHASE 3: Running Predictions")
    print("=" * 60)

    predictor = Predictor()
    predictions, congested = predictor.predict_all(collector)

    print(f"\n  Predictions for all (switch, port) pairs:")
    print(f"  {'Switch':<8} {'Port':<6} {'Predicted':>10} {'Status':>12}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*12}")

    for (sw, port), pred in sorted(predictions.items()):
        status = "⚠ CONGESTED" if pred >= UTILIZATION_THRESHOLD else "  OK"
        bar_len = int(pred * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {sw:<8} {port:<6} {pred:>9.1%}  {bar} {status}")

    if congested:
        print(f"\n  ⚠ {len(congested)} link(s) predicted to be congested!")
    else:
        print(f"\n  ✓ No congestion predicted.")

    return predictions, congested


# ======================================================================
#  PHASE 4: Demonstrate rerouting
# ======================================================================
def demonstrate_rerouting(congested):
    print("\n" + "=" * 60)
    print("  PHASE 4: Congestion-Aware Rerouting")
    print("=" * 60)

    engine = RoutingEngine()

    # Show default path h1 → h4
    default_path = engine.shortest_path("s1", "s4")
    print(f"\n  Default path h1 → h4:  {' → '.join(default_path)}")

    if congested:
        alt_path = engine.compute_alternate_path(
            "10.0.0.1", "10.0.0.4", congested
        )
        print(f"  Alternate path h1 → h4: {' → '.join(alt_path)}")
        print(f"\n  Flow actions to install:")
        actions = engine.path_to_flow_actions(alt_path)
        for a in actions:
            if a["next_hop"]:
                print(f"    Switch {a['switch']}: forward to {a['next_hop']}")
            else:
                print(f"    Switch {a['switch']}: deliver to host")

        print(f"\n  ✓ In production, the Ryu controller would install")
        print(f"    OFPFlowMod rules with priority=10, idle_timeout=30")
        print(f"    on each switch along this alternate path.")
    else:
        print("  No rerouting needed — all links within threshold.")

    # Show all available paths
    print(f"\n  All available paths h1 → h4:")
    for p in engine.all_paths("s1", "s4"):
        print(f"    {' → '.join(p)}")


# ======================================================================
#  MAIN
# ======================================================================
def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Proactive Congestion Avoidance in SDN                  ║")
    print("║  using Machine Learning Traffic Prediction              ║")
    print("║                                                         ║")
    print("║  Full Pipeline Simulation                               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Clean previous run
    for f in [TELEMETRY_CSV, MODEL_PATH]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Cleaned: {f}")

    collector = TelemetryCollector()
    csv_logger = DatasetLogger()

    # Phase 1
    generate_synthetic_data(collector, csv_logger, num_samples=200)

    # Phase 2
    train_model()

    # Phase 3
    predictions, congested = run_predictions(collector)

    # Phase 4
    demonstrate_rerouting(congested)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Telemetry records : {200 * 16}")
    print(f"  CSV file          : {TELEMETRY_CSV}")
    print(f"  Trained model     : {MODEL_PATH}")
    print(f"  Congested links   : {len(congested)}")
    print(f"  Threshold         : {UTILIZATION_THRESHOLD:.0%}")
    print(f"  Prediction horizon: 10 seconds ahead")
    print()
    print("  ✓ Pipeline complete. Ready for Mininet deployment.")
    print()


if __name__ == "__main__":
    main()
