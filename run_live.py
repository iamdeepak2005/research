#!/usr/bin/env python3
"""
run_live.py — SDN Simulation with dummy traffic & Dijkstra routing.

Simulates MULTIPLE traffic sources creating congestion scenarios.
Two-model architecture:
  1. Traffic Generator Model — produces realistic congestion patterns
     (multiple sources flooding a link, random bursts, cascade congestion)
  2. ML Predictor Model — trained online, predicts congestion BEFORE it happens
     and triggers Dijkstra rerouting proactively

Configurable:
  --routers N     Number of routers in the full-mesh topology (default: 4)
  --no-gui        Disable web dashboard, terminal only
  --port P        Web dashboard port (default: 8050)

Usage:
    python run_live.py                    # 4 routers
    python run_live.py --routers 6        # 6 routers
    python run_live.py --no-gui           # terminal only

Press Ctrl+C to stop at any time.
"""

import time
import random
import math
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    LINK_CAPACITY_BYTES, UTILIZATION_THRESHOLD, TELEMETRY_CSV,
    MODEL_PATH, now_iso, bytes_to_utilization, POLL_INTERVAL,
    generate_topology,
)
from telemetry_collector import TelemetryCollector
from dataset_logger import DatasetLogger
from predictor import Predictor
from routing_engine import RoutingEngine


# ── ANSI colors ──────────────────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(n_routers):
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║  SDN Proactive Congestion Avoidance — Multi-Source Sim      ║
║  Traffic Generator → ML Prediction → Dijkstra Rerouting    ║
║  Topology: {n_routers} routers, {n_routers} hosts, full mesh ({n_routers*(n_routers-1)//2} links)        ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


# ==================================================================
#  TRAFFIC GENERATOR MODEL
#  Simulates realistic multi-source congestion patterns
# ==================================================================
class TrafficGenerator:
    """
    Model 1: Generates realistic traffic that causes congestion.
    
    Simulates multiple hosts sending data simultaneously to create
    congestion. Models real-world scenarios:
      - Multiple sources flooding a single link
      - Cascade congestion (one congested link pushes traffic to another)
      - Random burst events (sudden spikes)
      - Gradual buildup from sustained transfers
    """

    def __init__(self, n_routers):
        self.n_routers = n_routers
        self.active_flows = []       # Currently active traffic flows
        self.congestion_events = []  # Scheduled congestion scenarios
        self.packet_collisions = []  # Active collision locations
        self._schedule_events()

    def _schedule_events(self):
        """Pre-schedule congestion scenarios throughout the simulation."""
        self.congestion_events = [
            # (start_time, duration, type, params)
            # Phase 1: Warm-up — light traffic on all links
            # Phase 2: Multi-source flood — 3+ hosts targeting same link
            {"start": 40, "duration": 50, "type": "multi_source_flood",
             "target_link": ("s1", 2), "n_sources": 3, "peak_util": 0.95},
            # Phase 3: Brief calm
            # Phase 4: Cascade congestion — flooding spills to neighbors
            {"start": 110, "duration": 40, "type": "cascade",
             "origin_link": ("s1", 2), "spread_links": [("s2", 3)],
             "peak_util": 0.92},
            # Phase 5: Random burst
            {"start": 170, "duration": 20, "type": "burst",
             "links": [("s1", 3), ("s1", 4)], "peak_util": 0.88},
            # Phase 6: Sustained multi-flow congestion
            {"start": 210, "duration": 60, "type": "multi_source_flood",
             "target_link": ("s2", 3), "n_sources": 4, "peak_util": 0.93},
            # Phase 7: Recovery + random spikes
            {"start": 290, "duration": 999, "type": "random_spikes",
             "spike_chance": 0.20, "spike_duration": 15, "peak_util": 0.90},
        ]

    def get_utilization(self, elapsed, sw, port):
        """
        Compute link utilization based on all active traffic patterns.
        Multiple sources can stack their contributions on the same link.
        """
        base_util = self._background_traffic(elapsed, sw, port)
        event_util = 0.0
        collision = False

        for event in self.congestion_events:
            start = event["start"]
            duration = event["duration"]
            if elapsed < start or elapsed > start + duration:
                continue

            progress = (elapsed - start) / duration
            etype = event["type"]

            if etype == "multi_source_flood":
                # Multiple hosts converging on target link
                tgt_sw, tgt_port = event["target_link"]
                if sw == tgt_sw and port == tgt_port:
                    n_src = event["n_sources"]
                    peak = event["peak_util"]
                    # Ramp up → sustain → ramp down
                    if progress < 0.2:
                        intensity = progress / 0.2
                    elif progress > 0.8:
                        intensity = (1.0 - progress) / 0.2
                    else:
                        intensity = 1.0
                    # Each source adds ~peak/n_sources with some noise
                    per_source = peak / n_src
                    event_util += sum(
                        per_source * intensity + random.gauss(0, 0.02)
                        for _ in range(n_src)
                    )
                    if event_util > UTILIZATION_THRESHOLD:
                        collision = True
                # Neighboring links also see elevated traffic
                elif sw == tgt_sw:
                    event_util += random.uniform(0.05, 0.15) * (1 - progress)

            elif etype == "cascade":
                # Congestion spreading from one link to its neighbors
                origin_sw, origin_port = event["origin_link"]
                peak = event["peak_util"]
                if sw == origin_sw and port == origin_port:
                    intensity = min(1.0, progress * 2)
                    event_util += peak * intensity + random.gauss(0, 0.03)
                    if event_util > UTILIZATION_THRESHOLD:
                        collision = True
                for spread_sw, spread_port in event["spread_links"]:
                    if sw == spread_sw and port == spread_port:
                        # Cascade delay
                        delayed = max(0, progress - 0.3) / 0.7
                        event_util += peak * 0.8 * delayed + random.gauss(0, 0.03)
                        if event_util > 0.7:
                            collision = True

            elif etype == "burst":
                # Sudden spike on multiple links
                for burst_sw, burst_port in event["links"]:
                    if sw == burst_sw and port == burst_port:
                        peak = event["peak_util"]
                        # Sharp spike shape
                        spike = math.exp(-((progress - 0.5) ** 2) / 0.05) * peak
                        event_util += spike + random.gauss(0, 0.02)
                        if spike > UTILIZATION_THRESHOLD:
                            collision = True

            elif etype == "random_spikes":
                # Recurring random congestion events
                spike_chance = event["spike_chance"]
                # Use deterministic seed per cycle for consistency
                cycle_id = int(elapsed / 10)
                rng = random.Random(cycle_id * 100 + port * 10 + hash(sw))
                if rng.random() < spike_chance:
                    spike_peak = event["peak_util"]
                    phase = (elapsed % 15) / 15.0
                    spike = math.sin(phase * math.pi) * spike_peak
                    event_util += spike
                    if spike > UTILIZATION_THRESHOLD:
                        collision = True

        total = min(1.0, max(0.0, base_util + event_util))
        return total, collision

    def _background_traffic(self, elapsed, sw, port):
        """Low-level background traffic with slight variation."""
        # Time-varying base with per-link personality
        sw_idx = int(sw[1:]) - 1
        seed = sw_idx * 100 + port
        phase = math.sin(elapsed * 0.05 + seed * 0.7) * 0.05
        base = 0.08 + phase + random.gauss(0, 0.02)
        return max(0.02, min(0.30, base))


def generate_dummy_records(traffic_gen, elapsed, switches_ports, n_routers):
    """Generate dummy telemetry with multi-source congestion patterns."""
    ts = now_iso()
    records = []
    collisions = []

    for sw, port in switches_ports:
        rate, has_collision = traffic_gen.get_utilization(elapsed, sw, port)
        tx_bytes = int(LINK_CAPACITY_BYTES * rate * POLL_INTERVAL)
        rx_bytes = int(tx_bytes * random.uniform(0.6, 0.9))

        if has_collision:
            collisions.append({"switch": sw, "port": port, "util": rate})

        records.append({
            "timestamp": ts,
            "switch_id": sw,
            "port_no": port,
            "tx_packets": max(1, tx_bytes // 1500),
            "rx_packets": max(1, rx_bytes // 1500),
            "tx_bytes": tx_bytes,
            "rx_bytes": rx_bytes,
            "duration_sec": POLL_INTERVAL,
        })
    return records, collisions


# ==================================================================
#  TERMINAL OUTPUT (--no-gui)
# ==================================================================
def make_bar(value, width=25):
    filled = int(value * width)
    empty = width - filled
    if value >= UTILIZATION_THRESHOLD:
        color = RED
    elif value >= 0.5:
        color = YELLOW
    else:
        color = GREEN
    return f"{color}{'█' * filled}{'░' * empty}{RESET}"


def print_status(cycle, elapsed, predictions, congested, reroute_active,
                 model_status, predict_ms=0, train_ms=0, key_links=None,
                 collisions=None):
    print(f"\n{DIM}{'─' * 65}{RESET}")
    timing = ""
    if predict_ms > 0:
        timing = f"  │  Pred: {predict_ms:.1f}ms"
    if train_ms > 0:
        timing += f"  │  Train: {train_ms:.0f}ms"
    print(f"  {BOLD}Cycle {cycle}{RESET}  │  "
          f"Time: {elapsed:.0f}s  │  "
          f"Model: {model_status}  │  "
          f"Reroute: {f'{RED}ACTIVE{RESET}' if reroute_active else f'{GREEN}idle{RESET}'}"
          f"{timing}")
    print(f"{DIM}{'─' * 65}{RESET}")

    if key_links is None:
        key_links = [("s1", 2), ("s1", 3), ("s1", 4), ("s2", 3)]
    for sw, port in key_links:
        pred = predictions.get((sw, port), 0)
        bar = make_bar(pred)
        label = f"{sw}:p{port}"
        flag = f" {RED}⚠ CONGESTED{RESET}" if (sw, port) in congested else ""
        print(f"  {label:<6} {bar} {pred:>6.1%}{flag}")

    if collisions:
        print(f"  {RED}{BOLD}💥 PACKET COLLISIONS: {len(collisions)} link(s){RESET}")
        for c in collisions:
            print(f"     {RED}{c['switch']}:p{c['port']} @ {c['util']:.0%}{RESET}")

    other_vals = [v for (s, p), v in predictions.items()
                  if (s, p) not in key_links]
    if other_vals:
        avg = sum(other_vals) / len(other_vals)
        mx = max(other_vals)
        print(f"  {DIM}other  avg={avg:.1%}  max={mx:.1%}  ({len(other_vals)} links){RESET}")


# ==================================================================
#  MAIN
# ==================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SDN Proactive Congestion Avoidance — Multi-Source Simulation"
    )
    parser.add_argument(
        "--routers", "--switches", type=int, default=4, dest="routers",
        help="Number of routers in the full-mesh topology (default: 4)"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Disable web dashboard, use terminal output only"
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port for the web dashboard (default: 8050)"
    )
    args = parser.parse_args()

    # ── Build topology ──
    n_routers = max(2, args.routers)
    topo = generate_topology(n_routers)

    switches_ports = topo["switches_ports"]

    banner(n_routers)
    print(f"  {CYAN}Mode: {BOLD}MULTI-SOURCE SIMULATION{RESET}")
    print(f"  {CYAN}Topology: {n_routers} routers, {len(topo['hosts'])} hosts, "
          f"{len(topo['switch_links'])} links (full mesh){RESET}")
    print(f"  {CYAN}Routing: Dijkstra's algorithm (utilization-weighted edges){RESET}")
    print(f"  {CYAN}Traffic: Multiple sources create congestion → ML detects → Dijkstra reroutes{RESET}")

    # ── Clean previous run FIRST (before logger so header is preserved) ──
    for f in [TELEMETRY_CSV, MODEL_PATH]:
        if os.path.exists(f):
            os.remove(f)

    # ── Initialize (logger creates fresh CSV with correct header) ──
    collector = TelemetryCollector()
    csv_logger = DatasetLogger()
    routing = RoutingEngine(topo_config=topo)
    traffic_gen = TrafficGenerator(n_routers)
    predictor = None
    model_trained = False
    reroute_active = False
    reroute_path_str = ""
    active_path_list = []

    # ── Dashboard ──
    dashboard = None
    if not args.no_gui:
        from web_dashboard import WebDashboard
        dashboard = WebDashboard(port=args.port, n_switches=n_routers, routing_engine=routing)
        dashboard.start(open_browser=True)
        dashboard.add_event(f"Dashboard started — {n_routers}-router topology (Dijkstra routing)")
        dashboard.add_event("Traffic: Multi-source congestion simulation active")
    else:
        print(f"  {DIM}GUI disabled — terminal output only{RESET}")

    # Default path via Dijkstra
    all_switches = topo["switches"]
    src_ip = list(topo["ip_host"].keys())[0] if topo["ip_host"] else "10.0.0.1"
    dst_ip = list(topo["ip_host"].keys())[-1] if len(topo["ip_host"]) > 1 else src_ip
    first_sw = all_switches[0]
    last_sw = all_switches[-1] if len(all_switches) > 1 else first_sw
    default_path = routing.shortest_path(first_sw, last_sw) if first_sw != last_sw else [first_sw]

    # Key links for terminal display
    key_links = [("s1", p) for p in range(2, min(n_routers + 1, 7))]
    if n_routers >= 3:
        key_links.append(("s2", 3))

    # Train faster with multi-source data (data is richer)
    min_train_cycles = 20
    n_ports_per_cycle = len(switches_ports)

    print(f"\n  {CYAN}Polling every {POLL_INTERVAL}s | Threshold: {UTILIZATION_THRESHOLD:.0%}{RESET}")
    print(f"  Model auto-trains after {min_train_cycles} cycles "
          f"(~{min_train_cycles * POLL_INTERVAL}s)")
    print(f"  Default path (Dijkstra): {' → '.join(default_path)}")
    print(f"  Congestion scenarios scheduled: multi-source flood, cascade, bursts")
    print(f"  Press Ctrl+C to stop\n")

    if dashboard:
        dashboard.add_event(f"Poll: {POLL_INTERVAL}s | Threshold: {UTILIZATION_THRESHOLD:.0%}")
        dashboard.add_event(f"Default path (Dijkstra): {' → '.join(default_path)}")
        dashboard.add_event(f"Congestion scenarios: flood, cascade, bursts, random spikes")

    start_time = time.time()
    cycle = 0
    predict_ms = 0
    train_ms = 0
    r2_val = None
    mae_val = None

    try:
        while True:
            cycle += 1
            elapsed = time.time() - start_time

            # ── Generate dummy telemetry with multi-source congestion ──
            records, collisions = generate_dummy_records(
                traffic_gen, elapsed, switches_ports, n_routers
            )

            for record in records:
                collector.add_record(record)
                csv_logger.log(record)

            # ── Compute throughput ──
            total_tx = sum(r["tx_bytes"] for r in records) * 8 / POLL_INTERVAL
            total_rx = sum(r["rx_bytes"] for r in records) * 8 / POLL_INTERVAL
            nic_tx_mbps = total_tx / 1e6
            nic_rx_mbps = total_rx / 1e6

            # ── Log collision events ──
            if collisions and dashboard:
                for c in collisions:
                    dashboard.add_event(
                        f"💥 Packet collision: {c['switch']}:p{c['port']} "
                        f"({c['util']:.0%} utilization — multiple sources)"
                    )

            # ── Auto-train model (faster with rich data) ──
            if not model_trained and cycle >= min_train_cycles:
                train_msg = f"Auto-training ML model ({cycle * n_ports_per_cycle} records)..."
                if dashboard:
                    dashboard.add_event(f"⏳ {train_msg}")
                print(f"  {YELLOW}{BOLD}>>> {train_msg}{RESET}")

                t0 = time.time()
                try:
                    from train_model import train as train_model_fn
                    train_model_fn()
                    train_ms = (time.time() - t0) * 1000

                    predictor = Predictor()
                    model_trained = True

                    try:
                        from train_model import load_data, engineer_features, FEATURE_COLS
                        from sklearn.metrics import r2_score, mean_absolute_error
                        import joblib
                        df = load_data(TELEMETRY_CSV)
                        df = engineer_features(df)
                        model = joblib.load(MODEL_PATH)
                        X = df[FEATURE_COLS].values
                        y = df["target"].values
                        y_pred = model.predict(X)
                        r2_val = r2_score(y, y_pred)
                        mae_val = mean_absolute_error(y, y_pred)
                    except Exception:
                        pass

                    trained_msg = f"Model trained in {train_ms:.0f}ms"
                    if r2_val is not None:
                        trained_msg += f" | R²={r2_val:.4f} | MAE={mae_val:.6f}"

                    if dashboard:
                        dashboard.add_event(f"✅ {trained_msg}")
                    print(f"  {GREEN}{BOLD}>>> {trained_msg}{RESET}")

                except Exception as e:
                    train_ms = (time.time() - t0) * 1000
                    err_msg = f"Training failed ({train_ms:.0f}ms): {str(e)[:80]}"
                    if dashboard:
                        dashboard.add_event(f"❌ {err_msg}")
                    print(f"  {RED}{BOLD}>>> {err_msg}{RESET}")
                    # Retry next cycle with more data
                    min_train_cycles = cycle + 5

            # ── Predict & reroute (Dijkstra) ──
            predictions = {}
            congested = []

            if model_trained and predictor:
                t0 = time.time()
                predictions, congested = predictor.predict_all(collector)
                predict_ms = (time.time() - t0) * 1000

                # Update Dijkstra link weights with latest utilizations
                routing.update_link_weights(predictions)

                if congested and not reroute_active:
                    alt_path = routing.compute_alternate_path(
                        src_ip, dst_ip, congested,
                        utilizations=predictions
                    )
                    reroute_active = True
                    active_path_list = alt_path
                    reroute_path_str = " → ".join(alt_path) if alt_path else ""

                    cong_summary = ', '.join(f'{s}:p{p}' for s, p in congested)
                    if dashboard:
                        dashboard.add_event(
                            f"🚨 CONGESTION PREDICTED on {cong_summary}"
                        )
                        for sw, port in congested:
                            dashboard.add_event(
                                f"  ⚠ {sw}:port{port} → predicted "
                                f"{predictions[(sw,port)]:.1%} (multiple sources)"
                            )
                        dashboard.add_event(f"  ⤷ Dijkstra reroute: {reroute_path_str}")
                    else:
                        print(f"\n  {RED}{BOLD}🚨 CONGESTION PREDICTED!{RESET}")
                        for sw, port in congested:
                            print(f"     {sw}:port{port} → predicted "
                                  f"{predictions[(sw,port)]:.1%}")
                        print(f"  {YELLOW}⤷ Dijkstra reroute: {reroute_path_str}{RESET}")

                    actions = routing.path_to_flow_actions(alt_path)
                    for a in actions:
                        msg = (f"FlowMod: {a['switch']} → output to {a['next_hop']}"
                               if a["next_hop"]
                               else f"FlowMod: {a['switch']} → deliver to host")
                        if dashboard:
                            dashboard.add_event(f"     {msg}")
                        else:
                            print(f"     {msg}")

                elif not congested and reroute_active:
                    reroute_active = False
                    active_path_list = default_path
                    default_str = " → ".join(default_path)

                    if dashboard:
                        dashboard.add_event("✓ Congestion cleared — traffic normalized!")
                        dashboard.add_event(f"  ⤷ Restoring default (Dijkstra): {default_str}")
                    else:
                        print(f"\n  {GREEN}{BOLD}✓ Congestion cleared!{RESET}")
                        print(f"  {GREEN}⤷ Default: {default_str}{RESET}")

                    reroute_path_str = ""

            # ── Model status ──
            if model_trained:
                model_status = "trained"
                model_status_term = f"{GREEN}trained (R²){RESET}"
            elif cycle >= min_train_cycles - 5:
                model_status = f"training in {min_train_cycles - cycle}..."
                model_status_term = f"{YELLOW}{model_status}{RESET}"
            else:
                model_status = f"collecting ({cycle}/{min_train_cycles})"
                model_status_term = f"{DIM}{model_status}{RESET}"

            # ── Push to dashboard ──
            if dashboard:
                pred_str = {f"{sw}:{port}": val
                            for (sw, port), val in predictions.items()}
                congested_str = [[sw, port] for sw, port in congested]
                collision_data = [
                    {"switch": c["switch"], "port": c["port"], "util": round(c["util"], 3)}
                    for c in collisions
                ]

                dashboard.update({
                    "cycle": cycle,
                    "elapsed": elapsed,
                    "mode": "simulate",
                    "model_status": model_status,
                    "reroute_active": reroute_active,
                    "reroute_path": reroute_path_str,
                    "train_time_ms": train_ms,
                    "predict_time_ms": predict_ms,
                    "r2_score": r2_val,
                    "mae_score": mae_val,
                    "predictions": pred_str,
                    "congested": congested_str,
                    "collisions": collision_data,
                    "nic_tx_mbps": nic_tx_mbps,
                    "nic_rx_mbps": nic_rx_mbps,
                    "default_path": default_path,
                    "active_path": active_path_list if reroute_active else default_path,
                })

            # ── Terminal output ──
            if args.no_gui or not dashboard:
                congested_set = set(congested)
                print_status(cycle, elapsed, predictions, congested_set,
                             reroute_active, model_status_term,
                             predict_ms=predict_ms, train_ms=train_ms,
                             key_links=key_links, collisions=collisions)

            # ── Wait ──
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n{CYAN}{BOLD}{'═' * 55}{RESET}")
        print(f"  {BOLD}Stopped after {elapsed:.0f}s ({cycle} cycles){RESET}")
        print(f"  Records: {cycle * n_ports_per_cycle}")
        print(f"  CSV: {TELEMETRY_CSV}")
        if model_trained:
            print(f"  Model: {MODEL_PATH}")
        print(f"  Routing: Dijkstra's algorithm")
        print(f"{CYAN}{BOLD}{'═' * 55}{RESET}\n")

        if dashboard:
            dashboard.add_event(f"🛑 Stopped after {elapsed:.0f}s ({cycle} cycles)")
            time.sleep(1)
            dashboard.stop()


if __name__ == "__main__":
    main()
