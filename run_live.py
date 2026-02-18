#!/usr/bin/env python3
"""
run_live.py — Live real-time SDN simulation.

This is NOT static. It runs continuously:
  - Every 5 seconds: generates new traffic telemetry (simulates switch stats)
  - After 30 samples: auto-trains the ML model
  - After training: predicts utilization in real-time every cycle
  - When congestion is predicted: triggers live rerouting

Traffic pattern changes over time to simulate a real network:
  Phase 1 (0-60s)   : Normal traffic on all links (~20-30%)
  Phase 2 (60-120s)  : Traffic ramp on s1→s2 link (increasing to 90%+)
  Phase 3 (120-180s) : Congestion detected → reroute triggered
  Phase 4 (180-240s) : Traffic drops back to normal (reroute clears)
  Phase 5 (240s+)    : Random bursts to show ongoing prediction

Press Ctrl+C to stop at any time.

Usage:
    python run_live.py
"""

import time
import random
import math
import sys
import os
import signal

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    LINK_CAPACITY_BYTES, UTILIZATION_THRESHOLD, TELEMETRY_CSV,
    MODEL_PATH, now_iso, bytes_to_utilization, POLL_INTERVAL,
)
from telemetry_collector import TelemetryCollector
from dataset_logger import DatasetLogger
from predictor import Predictor
from routing_engine import RoutingEngine


# ── ANSI colors for terminal output ──────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def clear_line():
    sys.stdout.write("\033[2K\r")


def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║  SDN Proactive Congestion Avoidance — LIVE Simulation       ║
║  Real-time telemetry → ML prediction → Dynamic rerouting    ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


def traffic_rate(elapsed, sw, port):
    """
    Returns a utilization fraction (0.0-1.0) based on elapsed time.
    Traffic patterns change dynamically to simulate real network behavior.
    """
    if sw == "s1" and port == 2:
        # s1→s2 link: the one that will get congested
        if elapsed < 60:
            # Phase 1: normal
            return random.uniform(0.15, 0.30)
        elif elapsed < 120:
            # Phase 2: ramp up
            progress = (elapsed - 60) / 60.0
            base = 0.30 + 0.65 * progress
            return min(1.0, base + random.gauss(0, 0.03))
        elif elapsed < 180:
            # Phase 3: sustained congestion
            return min(1.0, random.uniform(0.85, 0.97))
        elif elapsed < 240:
            # Phase 4: traffic drops
            progress = (elapsed - 180) / 60.0
            base = 0.90 - 0.65 * progress
            return max(0.05, base + random.gauss(0, 0.03))
        else:
            # Phase 5: random bursts
            if random.random() < 0.15:
                return random.uniform(0.75, 0.95)  # burst
            return random.uniform(0.10, 0.35)

    elif sw == "s2" and port == 3:
        # s2→s3: moderate traffic, slight increase
        base = 0.10 + 0.002 * elapsed
        return min(0.6, base + random.gauss(0, 0.03))

    else:
        # All other links: low random traffic
        return random.uniform(0.05, 0.25)


def make_bar(value, width=25):
    """Create a colored progress bar."""
    filled = int(value * width)
    empty = width - filled
    if value >= UTILIZATION_THRESHOLD:
        color = RED
    elif value >= 0.5:
        color = YELLOW
    else:
        color = GREEN
    return f"{color}{'█' * filled}{'░' * empty}{RESET}"


def print_status(cycle, elapsed, predictions, congested, reroute_active, model_status):
    """Print a compact live dashboard."""
    print(f"\n{DIM}{'─' * 65}{RESET}")
    print(f"  {BOLD}Cycle {cycle}{RESET}  │  "
          f"Time: {elapsed:.0f}s  │  "
          f"Model: {model_status}  │  "
          f"Reroute: {f'{RED}ACTIVE{RESET}' if reroute_active else f'{GREEN}idle{RESET}'}")
    print(f"{DIM}{'─' * 65}{RESET}")

    # Show key links
    key_links = [("s1", 2), ("s1", 3), ("s1", 4), ("s2", 3)]
    for sw, port in key_links:
        pred = predictions.get((sw, port), 0)
        bar = make_bar(pred)
        label = f"s{sw[-1]}:p{port}"
        flag = f" {RED}⚠ CONGESTED{RESET}" if (sw, port) in congested else ""
        print(f"  {label:<6} {bar} {pred:>6.1%}{flag}")

    # Show other links summary
    other_vals = [v for (s, p), v in predictions.items()
                  if (s, p) not in key_links]
    if other_vals:
        avg = sum(other_vals) / len(other_vals)
        mx = max(other_vals)
        print(f"  {DIM}other  avg={avg:.1%}  max={mx:.1%}  ({len(other_vals)} links){RESET}")


def main():
    banner()

    # Clean previous run
    for f in [TELEMETRY_CSV, MODEL_PATH]:
        if os.path.exists(f):
            os.remove(f)

    collector = TelemetryCollector()
    csv_logger = DatasetLogger()
    routing = RoutingEngine()
    predictor = None
    model_trained = False
    reroute_active = False
    reroute_path = None

    switches_ports = [
        ("s1", 1), ("s1", 2), ("s1", 3), ("s1", 4),
        ("s2", 1), ("s2", 2), ("s2", 3), ("s2", 4),
        ("s3", 1), ("s3", 2), ("s3", 3), ("s3", 4),
        ("s4", 1), ("s4", 2), ("s4", 3), ("s4", 4),
    ]

    print(f"  {CYAN}Starting live telemetry collection...{RESET}")
    print(f"  Polling every {POLL_INTERVAL}s | Threshold: {UTILIZATION_THRESHOLD:.0%}")
    print(f"  Model will auto-train after 30 cycles (~{30*POLL_INTERVAL}s)")
    print(f"  Press Ctrl+C to stop\n")

    start_time = time.time()
    cycle = 0

    try:
        while True:
            cycle += 1
            elapsed = time.time() - start_time
            ts = now_iso()

            # ── Generate telemetry for this cycle ──
            for sw, port in switches_ports:
                rate = traffic_rate(elapsed, sw, port)
                tx_bytes = int(LINK_CAPACITY_BYTES * rate * POLL_INTERVAL)
                rx_bytes = int(tx_bytes * random.uniform(0.6, 0.9))

                record = {
                    "timestamp": ts,
                    "switch_id": sw,
                    "port_no": port,
                    "tx_packets": max(1, tx_bytes // 1500),
                    "rx_packets": max(1, rx_bytes // 1500),
                    "tx_bytes": tx_bytes,
                    "rx_bytes": rx_bytes,
                    "duration_sec": POLL_INTERVAL,
                }
                collector.add_record(record)
                csv_logger.log(record)

            # ── Auto-train model after enough data ──
            if not model_trained and cycle >= 30:
                print(f"\n  {YELLOW}{BOLD}>>> Auto-training ML model with {cycle * 16} records...{RESET}")
                from train_model import train
                train()
                predictor = Predictor()
                model_trained = True
                print(f"  {GREEN}{BOLD}>>> Model trained and loaded!{RESET}")

            # ── Predict & reroute ──
            predictions = {}
            congested = []

            if model_trained and predictor:
                predictions, congested = predictor.predict_all(collector)

                if congested and not reroute_active:
                    # Congestion just detected!
                    reroute_path = routing.compute_alternate_path(
                        "10.0.0.1", "10.0.0.4", congested
                    )
                    reroute_active = True
                    print(f"\n  {RED}{BOLD}🚨 CONGESTION DETECTED!{RESET}")
                    for sw, port in congested:
                        print(f"     {sw}:port{port} → predicted {predictions[(sw,port)]:.1%}")
                    print(f"  {YELLOW}⤷ Rerouting h1→h4 via: {' → '.join(reroute_path)}{RESET}")
                    actions = routing.path_to_flow_actions(reroute_path)
                    for a in actions:
                        if a["next_hop"]:
                            print(f"     FlowMod: {a['switch']} → output to {a['next_hop']}")
                        else:
                            print(f"     FlowMod: {a['switch']} → deliver to host")

                elif not congested and reroute_active:
                    # Congestion cleared
                    reroute_active = False
                    default = routing.shortest_path("s1", "s4")
                    print(f"\n  {GREEN}{BOLD}✓ Congestion cleared!{RESET}")
                    print(f"  {GREEN}⤷ Restoring default path: {' → '.join(default)}{RESET}")

            # ── Display dashboard ──
            if model_trained:
                model_status = f"{GREEN}trained (R²){RESET}"
            elif cycle >= 25:
                model_status = f"{YELLOW}training in {30 - cycle}...{RESET}"
            else:
                model_status = f"{DIM}collecting ({cycle}/30){RESET}"

            congested_set = set(congested)
            print_status(cycle, elapsed, predictions, congested_set,
                         reroute_active, model_status)

            # ── Wait for next cycle ──
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n{CYAN}{BOLD}{'═' * 55}{RESET}")
        print(f"  {BOLD}Simulation stopped after {elapsed:.0f}s ({cycle} cycles){RESET}")
        print(f"  Records collected: {cycle * 16}")
        print(f"  CSV: {TELEMETRY_CSV}")
        if model_trained:
            print(f"  Model: {MODEL_PATH}")
        print(f"{CYAN}{BOLD}{'═' * 55}{RESET}\n")


if __name__ == "__main__":
    main()
