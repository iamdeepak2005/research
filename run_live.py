#!/usr/bin/env python3
"""
run_live.py — Live real-time SDN simulation with web dashboard.

Three modes:
  --mode simulate   (default) Scripted traffic phases with random numbers
  --mode live       Real NIC traffic captured via psutil
  --mode discover   Auto-discovers real network topology (ARP + ping + traceroute)
                    then monitors real NIC traffic

Configurable:
  --switches N      Number of switches for simulate/live modes (default: 4)
                    Ignored in discover mode (topology is auto-detected)

Features:
  - Web dashboard with animated network topology & packet routing
  - Smooth Chart.js graphs with hardware-accelerated rendering
  - Dynamic topology (N switches, N hosts, full-mesh links)
  - ML model training & prediction timing with R²/MAE
  - Congestion detection & reroute visualization

Usage:
    python run_live.py                         # 4 switches, simulate
    python run_live.py --switches 6            # 6 switches, simulate
    python run_live.py --mode live             # real NIC traffic, 4 switches
    python run_live.py --mode discover         # auto-discover real network!
    python run_live.py --switches 5 --mode live
    python run_live.py --no-gui                # terminal only

Press Ctrl+C to stop at any time.
"""

import time
import random
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


def banner(n_switches):
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║  SDN Proactive Congestion Avoidance — LIVE Simulation       ║
║  Real-time telemetry → ML prediction → Dynamic rerouting    ║
║  Topology: {n_switches} switches, {n_switches} hosts, full mesh ({n_switches*(n_switches-1)//2} links)        ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


# ==================================================================
#  SIMULATED TRAFFIC
# ==================================================================
def traffic_rate(elapsed, sw, port, n_switches):
    """
    Returns utilization fraction (0.0-1.0) based on elapsed time.
    The s1→s2 link (s1, port 2) is the one that gets congested.
    Other links have varied background traffic.
    """
    if sw == "s1" and port == 2:
        # s1→s2: the congestion link
        if elapsed < 60:
            return random.uniform(0.15, 0.30)
        elif elapsed < 120:
            progress = (elapsed - 60) / 60.0
            base = 0.30 + 0.65 * progress
            return min(1.0, base + random.gauss(0, 0.03))
        elif elapsed < 180:
            return min(1.0, random.uniform(0.85, 0.97))
        elif elapsed < 240:
            progress = (elapsed - 180) / 60.0
            base = 0.90 - 0.65 * progress
            return max(0.05, base + random.gauss(0, 0.03))
        else:
            if random.random() < 0.15:
                return random.uniform(0.75, 0.95)
            return random.uniform(0.10, 0.35)

    elif sw == "s2" and port == 3 and n_switches >= 3:
        # s2→s3: moderate with slow growth
        base = 0.10 + 0.002 * elapsed
        return min(0.6, base + random.gauss(0, 0.03))

    elif sw == "s1" and port == 3 and n_switches >= 3:
        # s1→s3: slightly elevated
        return random.uniform(0.08, 0.30)

    else:
        # Everything else: low random
        return random.uniform(0.05, 0.25)


def generate_simulated_records(elapsed, switches_ports, n_switches):
    """Generate simulated telemetry records."""
    ts = now_iso()
    records = []
    for sw, port in switches_ports:
        rate = traffic_rate(elapsed, sw, port, n_switches)
        tx_bytes = int(LINK_CAPACITY_BYTES * rate * POLL_INTERVAL)
        rx_bytes = int(tx_bytes * random.uniform(0.6, 0.9))
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
    return records


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
                 model_status, predict_ms=0, train_ms=0, key_links=None):
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
        description="SDN Proactive Congestion Avoidance — Live Runner"
    )
    parser.add_argument(
        "--mode", choices=["simulate", "live", "discover"], default="simulate",
        help="Traffic source: 'simulate' (random), 'live' (real NIC), or 'discover' (auto-detect network)"
    )
    parser.add_argument(
        "--switches", type=int, default=4,
        help="Number of switches in the full-mesh topology (default: 4)"
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

    # ── Build or discover topology ──
    if args.mode == "discover":
        from network_discovery import NetworkDiscovery
        disco = NetworkDiscovery(ping_timeout_ms=400, max_tracert_hops=6)
        topo = disco.discover()
        n_sw = topo["n_switches"]
        mode_label = "DISCOVERED (real network)"
    else:
        n_sw = max(2, args.switches)
        topo = generate_topology(n_sw)
        mode_label = "REAL NIC TRAFFIC (psutil)" if args.mode == "live" else "SIMULATED"

    switches_ports = topo["switches_ports"]

    banner(n_sw)
    print(f"  {CYAN}Mode: {BOLD}{mode_label}{RESET}")
    topo_type = "discovered" if topo.get("is_discovered") else "full mesh"
    print(f"  {CYAN}Topology: {n_sw} switches, {len(topo['hosts'])} hosts, "
          f"{len(topo['switch_links'])} links ({topo_type}){RESET}")

    # ── Initialize routing engine first (needed by dashboard editor) ──
    collector = TelemetryCollector()
    csv_logger = DatasetLogger()
    routing = RoutingEngine(topo_config=topo)
    predictor = None
    model_trained = False
    reroute_active = False
    reroute_path_str = ""
    active_path_list = []

    # ── Dashboard ──
    dashboard = None
    if not args.no_gui:
        from web_dashboard import WebDashboard
        dashboard = WebDashboard(port=args.port, n_switches=n_sw, routing_engine=routing)
        dashboard.start(open_browser=True)
        if topo.get("is_discovered"):
            dashboard.add_event(f"Network discovered — {n_sw} routers, {len(topo['hosts'])} devices")
            # Send device info for rich labels
            di = topo.get("device_info", {})
            for s in di.get("switches", []):
                dashboard.add_event(f"  Router {s['id'].upper()}: {s['ip']} ({s.get('hostname') or s.get('role')})")
            for h in di.get("hosts", []):
                dashboard.add_event(f"  Host {h['id'].upper()}: {h['ip']} ({h.get('hostname') or ''})")
        else:
            dashboard.add_event(f"Dashboard started — {n_sw}-switch topology (Dijkstra routing)")
    else:
        print(f"  {DIM}GUI disabled — terminal output only{RESET}")

    # ── NIC monitor (for live or discover modes) ──
    nic_monitor = None
    if args.mode in ("live", "discover"):
        from nic_monitor import NICMonitor
        nic_monitor = NICMonitor(switches_ports=switches_ports)
        if dashboard:
            dashboard.add_event(f"NIC Monitor: {len(nic_monitor.nic_names)} interface(s)")
            for nic in nic_monitor.nic_names:
                dashboard.add_event(f"  Monitoring: {nic}")

    # ── Clean previous run ──
    for f in [TELEMETRY_CSV, MODEL_PATH]:
        if os.path.exists(f):
            os.remove(f)

    # Default path
    all_hosts = topo["hosts"]
    all_switches = topo["switches"]
    # Source = first host IP, dest = last host IP
    src_ip = list(topo["ip_host"].keys())[0] if topo["ip_host"] else "10.0.0.1"
    dst_ip = list(topo["ip_host"].keys())[-1] if len(topo["ip_host"]) > 1 else src_ip
    first_sw = all_switches[0]
    last_sw = all_switches[-1] if len(all_switches) > 1 else first_sw
    default_path = routing.shortest_path(first_sw, last_sw) if first_sw != last_sw else [first_sw]

    # Key links for terminal display
    key_links = [("s1", p) for p in range(2, n_sw + 1)]
    if n_sw >= 3:
        key_links.append(("s2", 3))

    min_train_cycles = 30
    n_ports_per_cycle = len(switches_ports)

    print(f"\n  {CYAN}Polling every {POLL_INTERVAL}s | Threshold: {UTILIZATION_THRESHOLD:.0%}{RESET}")
    print(f"  Model auto-trains after {min_train_cycles} cycles "
          f"(~{min_train_cycles * POLL_INTERVAL}s)")
    print(f"  Default path: {' → '.join(default_path)}")
    print(f"  Press Ctrl+C to stop\n")

    if dashboard:
        dashboard.add_event(f"Mode: {args.mode.upper()} | Poll: {POLL_INTERVAL}s | "
                            f"Threshold: {UTILIZATION_THRESHOLD:.0%}")
        dashboard.add_event(f"Default path: {' → '.join(default_path)}")

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

            # ── Generate / capture telemetry ──
            if args.mode in ("live", "discover") and nic_monitor:
                records = nic_monitor.poll()
            else:
                records = generate_simulated_records(elapsed, switches_ports, n_sw)

            for record in records:
                collector.add_record(record)
                csv_logger.log(record)

            # ── NIC throughput ──
            nic_tx_mbps = 0
            nic_rx_mbps = 0
            if nic_monitor:
                rates = nic_monitor.get_raw_rates()
                nic_tx_mbps = sum(r.get("tx_bps", 0) for r in rates.values()) / 1e6
                nic_rx_mbps = sum(r.get("rx_bps", 0) for r in rates.values()) / 1e6
            else:
                total_tx = sum(r["tx_bytes"] for r in records) * 8 / POLL_INTERVAL
                total_rx = sum(r["rx_bytes"] for r in records) * 8 / POLL_INTERVAL
                nic_tx_mbps = total_tx / 1e6
                nic_rx_mbps = total_rx / 1e6

            # ── Auto-train model ──
            if not model_trained and cycle >= min_train_cycles:
                train_msg = f"Auto-training ML model ({cycle * n_ports_per_cycle} records)..."
                if dashboard:
                    dashboard.add_event(f"⏳ {train_msg}")
                print(f"  {YELLOW}{BOLD}>>> {train_msg}{RESET}")

                t0 = time.time()
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

            # ── Predict & reroute ──
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

                    if dashboard:
                        dashboard.add_event(
                            f"🚨 CONGESTION DETECTED on "
                            f"{', '.join(f'{s}:p{p}' for s, p in congested)}"
                        )
                        for sw, port in congested:
                            dashboard.add_event(
                                f"  ⚠ {sw}:port{port} → predicted "
                                f"{predictions[(sw,port)]:.1%}"
                            )
                        dashboard.add_event(f"  ⤷ Rerouting via: {reroute_path_str}")
                    else:
                        print(f"\n  {RED}{BOLD}🚨 CONGESTION DETECTED!{RESET}")
                        for sw, port in congested:
                            print(f"     {sw}:port{port} → predicted "
                                  f"{predictions[(sw,port)]:.1%}")
                        print(f"  {YELLOW}⤷ Rerouting via: {reroute_path_str}{RESET}")

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
                        dashboard.add_event("✓ Congestion cleared!")
                        dashboard.add_event(f"  ⤷ Restoring default: {default_str}")
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

                # Include device info if discovered
                device_info = topo.get("device_info")

                dashboard.update({
                    "cycle": cycle,
                    "elapsed": elapsed,
                    "mode": args.mode,
                    "model_status": model_status,
                    "reroute_active": reroute_active,
                    "reroute_path": reroute_path_str,
                    "train_time_ms": train_ms,
                    "predict_time_ms": predict_ms,
                    "r2_score": r2_val,
                    "mae_score": mae_val,
                    "predictions": pred_str,
                    "congested": congested_str,
                    "nic_tx_mbps": nic_tx_mbps,
                    "nic_rx_mbps": nic_rx_mbps,
                    "default_path": default_path,
                    "active_path": active_path_list if reroute_active else default_path,
                    "device_info": device_info,
                })

            # ── Terminal output ──
            if args.no_gui or not dashboard:
                congested_set = set(congested)
                print_status(cycle, elapsed, predictions, congested_set,
                             reroute_active, model_status_term,
                             predict_ms=predict_ms, train_ms=train_ms,
                             key_links=key_links)

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
        print(f"{CYAN}{BOLD}{'═' * 55}{RESET}\n")

        if dashboard:
            dashboard.add_event(f"🛑 Stopped after {elapsed:.0f}s ({cycle} cycles)")
            time.sleep(1)
            dashboard.stop()


if __name__ == "__main__":
    main()
