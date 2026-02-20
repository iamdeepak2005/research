#!/usr/bin/env python3
"""
nic_monitor.py — Real network traffic capture using psutil.

Reads actual NIC (Network Interface Card) byte counters from the OS,
computes deltas between polls, and maps real interfaces to virtual
SDN switch ports so the rest of the pipeline works unchanged.

Usage:
    from nic_monitor import NICMonitor
    monitor = NICMonitor()
    records = monitor.poll()   # returns list of telemetry dicts

Standalone test:
    python nic_monitor.py
"""

import time
import psutil
from utils import (
    LINK_CAPACITY_BYTES, POLL_INTERVAL, now_iso, bytes_to_utilization,
)


# ── NIC-to-virtual-switch mapping ────────────────────────────────────
# We map real Windows NIC names to virtual (switch, port) pairs.
# This lets us feed real traffic into the same pipeline that expects
# switch-level telemetry.
#
# Strategy:
#   - Auto-detect active NICs (those with traffic)
#   - Spread them across 4 virtual switches (s1-s4), 4 ports each
#   - If fewer NICs than 16 ports, the same NIC is mapped to multiple ports
#     with a random jitter so each virtual link looks slightly different

DEFAULT_NIC_MAP = None  # auto-detect


class NICMonitor:
    """
    Captures real network traffic from the OS using psutil.

    Each call to poll() returns a list of telemetry records (one per
    virtual switch-port) with real tx/rx byte deltas measured from
    the actual NIC counters.
    """

    def __init__(self, poll_interval=None, nic_names=None, switches_ports=None):
        """
        Args:
            poll_interval: seconds between intended polls (for rate calc)
            nic_names: list of NIC names to monitor. None = auto-detect.
            switches_ports: list of (switch_id, port_no) tuples. None = default 4×4.
        """
        self.poll_interval = poll_interval or POLL_INTERVAL
        self.nic_names = nic_names or self._detect_active_nics()
        self._prev_counters = {}   # {nic_name: (bytes_sent, bytes_recv, ts)}
        self._nic_port_map = {}    # {(switch, port): nic_name}
        self._switches_ports = switches_ports or [
            ("s1", 1), ("s1", 2), ("s1", 3), ("s1", 4),
            ("s2", 1), ("s2", 2), ("s2", 3), ("s2", 4),
            ("s3", 1), ("s3", 2), ("s3", 3), ("s3", 4),
            ("s4", 1), ("s4", 2), ("s4", 3), ("s4", 4),
        ]

        # Build the mapping of virtual ports → real NICs
        self._build_port_map()

        # Take initial snapshot so first poll has a delta
        self._snapshot()

        print(f"[NICMonitor] Monitoring {len(self.nic_names)} interface(s): "
              f"{', '.join(self.nic_names)}")
        print(f"[NICMonitor] Mapped to {len(self._nic_port_map)} virtual switch-ports")

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_active_nics():
        """
        Auto-detect NICs that are UP and have sent/received traffic.
        Excludes loopback and virtual adapters where possible.
        """
        stats = psutil.net_if_stats()
        counters = psutil.net_io_counters(pernic=True)
        active = []

        # Skip these common virtual/loopback names on Windows
        skip_keywords = ["loopback", "isatap", "teredo", "vethernet",
                         "vmware", "virtualbox", "docker", "wsl"]

        for nic_name, nic_stat in stats.items():
            lower = nic_name.lower()
            # Skip loopback / virtual
            if any(kw in lower for kw in skip_keywords):
                continue
            # Must be UP
            if not nic_stat.isup:
                continue
            # Must have some traffic
            nic_io = counters.get(nic_name)
            if nic_io and (nic_io.bytes_sent > 0 or nic_io.bytes_recv > 0):
                active.append(nic_name)

        if not active:
            # Fallback: use all NICs that are up
            active = [n for n, s in stats.items() if s.isup]

        if not active:
            # Last resort: use everything
            active = list(counters.keys())

        return active[:8]  # cap at 8 NICs

    # ------------------------------------------------------------------
    def _build_port_map(self):
        """
        Map virtual (switch, port) pairs to real NIC names.
        Virtual ports are filled round-robin from the available NICs.
        """
        for i, (sw, port) in enumerate(self._switches_ports):
            nic = self.nic_names[i % len(self.nic_names)]
            self._nic_port_map[(sw, port)] = nic

    # ------------------------------------------------------------------
    def _snapshot(self):
        """Take a snapshot of current NIC counters."""
        counters = psutil.net_io_counters(pernic=True)
        now = time.time()
        for nic in self.nic_names:
            io = counters.get(nic)
            if io:
                self._prev_counters[nic] = (
                    io.bytes_sent, io.bytes_recv,
                    io.packets_sent, io.packets_recv,
                    now,
                )

    # ------------------------------------------------------------------
    def poll(self):
        """
        Read current NIC counters, compute deltas from the last poll,
        and return a list of telemetry records in the standard format.

        Returns:
            list of dicts, one per virtual (switch, port), with keys:
                timestamp, switch_id, port_no, tx_packets, rx_packets,
                tx_bytes, rx_bytes, duration_sec
        """
        counters = psutil.net_io_counters(pernic=True)
        now = time.time()
        ts = now_iso()
        records = []

        # Compute per-NIC deltas
        nic_deltas = {}
        for nic in self.nic_names:
            io = counters.get(nic)
            prev = self._prev_counters.get(nic)
            if io and prev:
                prev_sent, prev_recv, prev_ps, prev_pr, prev_ts = prev
                dt = max(now - prev_ts, 0.01)
                nic_deltas[nic] = {
                    "tx_bytes": max(0, io.bytes_sent - prev_sent),
                    "rx_bytes": max(0, io.bytes_recv - prev_recv),
                    "tx_packets": max(0, io.packets_sent - prev_ps),
                    "rx_packets": max(0, io.packets_recv - prev_pr),
                    "duration_sec": round(dt, 2),
                }
            else:
                nic_deltas[nic] = {
                    "tx_bytes": 0, "rx_bytes": 0,
                    "tx_packets": 0, "rx_packets": 0,
                    "duration_sec": self.poll_interval,
                }

        # Map to virtual switch-ports
        for (sw, port), nic in self._nic_port_map.items():
            delta = nic_deltas.get(nic, {})
            # Distribute traffic across ports mapped to same NIC
            # by dividing evenly (how many ports share this NIC?)
            share_count = sum(
                1 for n in self._nic_port_map.values() if n == nic
            )
            record = {
                "timestamp": ts,
                "switch_id": sw,
                "port_no": port,
                "tx_packets": max(1, delta.get("tx_packets", 0) // share_count),
                "rx_packets": max(1, delta.get("rx_packets", 0) // share_count),
                "tx_bytes": delta.get("tx_bytes", 0) // share_count,
                "rx_bytes": delta.get("rx_bytes", 0) // share_count,
                "duration_sec": delta.get("duration_sec", self.poll_interval),
            }
            records.append(record)

        # Update snapshot for next delta
        for nic in self.nic_names:
            io = counters.get(nic)
            if io:
                self._prev_counters[nic] = (
                    io.bytes_sent, io.bytes_recv,
                    io.packets_sent, io.packets_recv,
                    now,
                )

        return records

    # ------------------------------------------------------------------
    def get_raw_rates(self):
        """
        Get current raw NIC throughput rates (bytes/sec) for display.
        Returns dict: {nic_name: {"tx_bps": float, "rx_bps": float}}
        """
        counters = psutil.net_io_counters(pernic=True)
        now = time.time()
        rates = {}

        for nic in self.nic_names:
            io = counters.get(nic)
            prev = self._prev_counters.get(nic)
            if io and prev:
                prev_sent, prev_recv, _, _, prev_ts = prev
                dt = max(now - prev_ts, 0.01)
                rates[nic] = {
                    "tx_bps": max(0, (io.bytes_sent - prev_sent) * 8 / dt),
                    "rx_bps": max(0, (io.bytes_recv - prev_recv) * 8 / dt),
                    "tx_bytes_total": io.bytes_sent,
                    "rx_bytes_total": io.bytes_recv,
                }
            else:
                rates[nic] = {
                    "tx_bps": 0, "rx_bps": 0,
                    "tx_bytes_total": 0, "rx_bytes_total": 0,
                }
        return rates


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  NIC Monitor — Live Test")
    print("=" * 60)

    mon = NICMonitor()

    print(f"\nDetected NICs: {mon.nic_names}")
    print(f"Port map: {mon._nic_port_map}\n")

    print("Waiting 3 seconds to measure traffic delta...\n")
    time.sleep(3)

    records = mon.poll()
    print(f"Got {len(records)} telemetry records:\n")
    for r in records:
        util = bytes_to_utilization(r["tx_bytes"], r["duration_sec"])
        print(f"  {r['switch_id']}:port{r['port_no']}  "
              f"tx={r['tx_bytes']:>10,} B  "
              f"rx={r['rx_bytes']:>10,} B  "
              f"util={util:.2%}")

    print("\n--- Raw NIC rates ---")
    rates = mon.get_raw_rates()
    for nic, r in rates.items():
        print(f"  {nic}: TX={r['tx_bps']/1e6:.2f} Mbps  "
              f"RX={r['rx_bps']/1e6:.2f} Mbps")
