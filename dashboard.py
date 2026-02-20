#!/usr/bin/env python3
"""
dashboard.py — Animated GUI dashboard for SDN Congestion Avoidance.

Displays real-time:
  - Link utilization time-series graphs
  - Congestion status and reroute events
  - ML model timing (train, predict)
  - NIC raw throughput
  - System event log

Uses matplotlib with a dark theme for a professional research look.
Called from run_live.py; do NOT run standalone.

Dependencies:
    pip install matplotlib psutil numpy
"""

import time
import collections
import threading
import matplotlib
matplotlib.use("TkAgg")  # Interactive backend for Windows

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import numpy as np

from utils import UTILIZATION_THRESHOLD


# ── Color palette ────────────────────────────────────────────────────
COLORS = {
    "bg":           "#0d1117",
    "panel":        "#161b22",
    "border":       "#30363d",
    "text":         "#e6edf3",
    "text_dim":     "#8b949e",
    "green":        "#3fb950",
    "green_dim":    "#238636",
    "yellow":       "#d29922",
    "red":          "#f85149",
    "red_dim":      "#da3633",
    "blue":         "#58a6ff",
    "cyan":         "#39d2c0",
    "purple":       "#bc8cff",
    "orange":       "#f0883e",
    "line_s1p2":    "#58a6ff",   # s1:port2 — the key link
    "line_s1p3":    "#3fb950",
    "line_s1p4":    "#bc8cff",
    "line_s2p3":    "#f0883e",
    "grid":         "#21262d",
    "threshold":    "#f8514966",
}

# Link colors mapping
LINK_COLORS = {
    ("s1", 2): COLORS["line_s1p2"],
    ("s1", 3): COLORS["line_s1p3"],
    ("s1", 4): COLORS["line_s1p4"],
    ("s2", 3): COLORS["line_s2p3"],
}

# Max data points to keep in rolling window
MAX_HISTORY = 120


class LiveDashboard:
    """
    Animated matplotlib dashboard for SDN live monitoring.

    Usage:
        dashboard = LiveDashboard()
        # From your main loop, push data:
        dashboard.push_utilization(cycle, predictions)
        dashboard.push_nic_rates(rates_dict)
        dashboard.push_event("Congestion detected on s1:p2!")
        dashboard.set_timing(train_ms=120, predict_ms=5)
        # Start the GUI (blocks main thread):
        dashboard.run()
    """

    def __init__(self):
        # Rolling data stores
        self._util_history = collections.defaultdict(
            lambda: collections.deque(maxlen=MAX_HISTORY)
        )
        self._time_history = collections.deque(maxlen=MAX_HISTORY)
        self._nic_tx = collections.deque(maxlen=MAX_HISTORY)
        self._nic_rx = collections.deque(maxlen=MAX_HISTORY)
        self._nic_time = collections.deque(maxlen=MAX_HISTORY)

        # Current state
        self._cycle = 0
        self._elapsed = 0.0
        self._congested = set()
        self._reroute_active = False
        self._reroute_path = ""
        self._model_status = "collecting"
        self._train_time_ms = 0.0
        self._predict_time_ms = 0.0
        self._r2_score = None
        self._mae_score = None
        self._events = collections.deque(maxlen=15)

        # NIC info
        self._nic_rates = {}

        # Thread lock for data updates
        self._lock = threading.Lock()

        # Figure and axes (created in run())
        self.fig = None
        self._axes = {}
        self._ani = None

    # ==================================================================
    #  PUBLIC API — call these from your main loop thread
    # ==================================================================

    def push_utilization(self, cycle, elapsed, predictions, congested=None,
                         reroute_active=False, reroute_path=""):
        """Push utilization predictions for the current cycle."""
        with self._lock:
            self._cycle = cycle
            self._elapsed = elapsed
            self._time_history.append(elapsed)
            self._congested = set(congested or [])
            self._reroute_active = reroute_active
            self._reroute_path = reroute_path

            for (sw, port), val in predictions.items():
                self._util_history[(sw, port)].append(val)

    def push_nic_rates(self, rates):
        """Push raw NIC throughput rates."""
        with self._lock:
            self._nic_rates = dict(rates)
            # Aggregate all NICs
            total_tx = sum(r.get("tx_bps", 0) for r in rates.values())
            total_rx = sum(r.get("rx_bps", 0) for r in rates.values())
            self._nic_tx.append(total_tx / 1e6)  # Mbps
            self._nic_rx.append(total_rx / 1e6)
            self._nic_time.append(self._elapsed)

    def push_event(self, message):
        """Add an event to the log."""
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            self._events.appendleft(f"[{ts}] {message}")

    def set_timing(self, train_ms=None, predict_ms=None):
        """Set ML timing values."""
        with self._lock:
            if train_ms is not None:
                self._train_time_ms = train_ms
            if predict_ms is not None:
                self._predict_time_ms = predict_ms

    def set_model_status(self, status, r2=None, mae=None):
        """Set model training status."""
        with self._lock:
            self._model_status = status
            if r2 is not None:
                self._r2_score = r2
            if mae is not None:
                self._mae_score = mae

    # ==================================================================
    #  BUILD DASHBOARD LAYOUT
    # ==================================================================

    def _setup_figure(self):
        """Create the figure, grid, and all axes."""
        plt.rcParams.update({
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": COLORS["border"],
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text_dim"],
            "ytick.color": COLORS["text_dim"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.5,
            "font.family": "Segoe UI",
            "font.size": 9,
        })

        self.fig = plt.figure(
            figsize=(16, 9),
            num="SDN Proactive Congestion Avoidance — Live Dashboard",
        )

        # Grid: 3 rows × 4 cols
        gs = gridspec.GridSpec(
            3, 4,
            figure=self.fig,
            hspace=0.35,
            wspace=0.3,
            left=0.06, right=0.97,
            top=0.90, bottom=0.06,
        )

        # -- Title banner --
        self.fig.text(
            0.5, 0.96,
            "SDN PROACTIVE CONGESTION AVOIDANCE",
            ha="center", va="center",
            fontsize=16, fontweight="bold",
            color=COLORS["cyan"],
            fontfamily="Segoe UI",
        )
        self._title_sub = self.fig.text(
            0.5, 0.925,
            "● LIVE  |  Cycle 0  |  0.0s elapsed",
            ha="center", va="center",
            fontsize=10, color=COLORS["text_dim"],
        )

        # -- Axes --
        # [0,0..2] Link utilization (large, top-left)
        self._axes["util"] = self.fig.add_subplot(gs[0, 0:3])
        # [0,3] Status panel
        self._axes["status"] = self.fig.add_subplot(gs[0, 3])
        # [1,0..2] NIC throughput
        self._axes["nic"] = self.fig.add_subplot(gs[1, 0:3])
        # [1,3] Timing panel
        self._axes["timing"] = self.fig.add_subplot(gs[1, 3])
        # [2,0..3] Event log
        self._axes["events"] = self.fig.add_subplot(gs[2, 0:3])
        # [2,3] Topology / gauges
        self._axes["topo"] = self.fig.add_subplot(gs[2, 3])

        # Initialize each panel
        self._init_util_ax()
        self._init_nic_ax()
        self._init_status_ax()
        self._init_timing_ax()
        self._init_events_ax()
        self._init_topo_ax()

    # ------------------------------------------------------------------
    def _init_util_ax(self):
        ax = self._axes["util"]
        ax.set_title("LINK UTILIZATION  (predicted)", fontsize=11,
                     fontweight="bold", color=COLORS["text"], pad=8)
        ax.set_ylabel("Utilization", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.3)
        # Threshold line
        ax.axhline(y=UTILIZATION_THRESHOLD, color=COLORS["red"],
                    linestyle="--", alpha=0.5, linewidth=1.5)
        ax.text(2, UTILIZATION_THRESHOLD + 0.03, f"Threshold ({UTILIZATION_THRESHOLD:.0%})",
                fontsize=8, color=COLORS["red"], alpha=0.7)
        # Fill danger zone
        ax.axhspan(UTILIZATION_THRESHOLD, 1.1, alpha=0.05, color=COLORS["red"])

    def _init_nic_ax(self):
        ax = self._axes["nic"]
        ax.set_title("REAL NIC THROUGHPUT", fontsize=11,
                     fontweight="bold", color=COLORS["text"], pad=8)
        ax.set_ylabel("Mbps", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylim(0, 10)
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.3)

    def _init_status_ax(self):
        ax = self._axes["status"]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("STATUS", fontsize=11, fontweight="bold",
                     color=COLORS["text"], pad=8)

    def _init_timing_ax(self):
        ax = self._axes["timing"]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("ML METRICS", fontsize=11, fontweight="bold",
                     color=COLORS["text"], pad=8)

    def _init_events_ax(self):
        ax = self._axes["events"]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("EVENT LOG", fontsize=11, fontweight="bold",
                     color=COLORS["text"], pad=8)

    def _init_topo_ax(self):
        ax = self._axes["topo"]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("LINK GAUGES", fontsize=11, fontweight="bold",
                     color=COLORS["text"], pad=8)

    # ==================================================================
    #  UPDATE (called every animation frame)
    # ==================================================================

    def _update(self, frame):
        """Main animation update — redraws all panels."""
        with self._lock:
            self._update_subtitle()
            self._update_util_plot()
            self._update_nic_plot()
            self._update_status_panel()
            self._update_timing_panel()
            self._update_events_panel()
            self._update_topo_panel()

    # ------------------------------------------------------------------
    def _update_subtitle(self):
        mode_dot = "●" if self._reroute_active else "●"
        mode_color = COLORS["red"] if self._reroute_active else COLORS["green"]
        status = "REROUTING" if self._reroute_active else "LIVE"
        self._title_sub.set_text(
            f"● {status}  |  Cycle {self._cycle}  |  "
            f"{self._elapsed:.0f}s elapsed  |  "
            f"Model: {self._model_status}"
        )
        self._title_sub.set_color(mode_color if self._reroute_active else COLORS["text_dim"])

    # ------------------------------------------------------------------
    def _update_util_plot(self):
        ax = self._axes["util"]
        # Clear existing lines but keep grid/threshold
        for line in ax.get_lines():
            if line.get_linestyle() != "--":
                line.remove()
        # Remove old fills (keep threshold fill)
        for coll in ax.collections[:]:
            if coll.get_alpha() != 0.05:
                coll.remove()

        times = list(self._time_history)
        if not times:
            return

        # Key links
        key_links = [("s1", 2), ("s1", 3), ("s1", 4), ("s2", 3)]
        labels = ["s1→s2", "s1→s3", "s1→s4", "s2→s3"]

        for (sw, port), label in zip(key_links, labels):
            data = list(self._util_history.get((sw, port), []))
            if not data:
                continue
            t = times[-len(data):]
            color = LINK_COLORS.get((sw, port), COLORS["text_dim"])

            is_congested = (sw, port) in self._congested
            lw = 2.5 if is_congested else 1.5
            alpha = 1.0 if is_congested else 0.8

            ax.plot(t, data, color=color, linewidth=lw, alpha=alpha,
                    label=label, solid_capstyle="round")

            # Glow for congested links
            if is_congested:
                ax.fill_between(t, data, alpha=0.15, color=color)

        # Auto-scale x-axis
        if times:
            xmin = max(0, times[-1] - 120)
            xmax = times[-1] + 5
            ax.set_xlim(xmin, xmax)

        # Legend
        handles = ax.get_legend_handles_labels()
        if handles[0]:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.3,
                     facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                     labelcolor=COLORS["text"])

    # ------------------------------------------------------------------
    def _update_nic_plot(self):
        ax = self._axes["nic"]
        ax.clear()
        self._init_nic_ax()

        tx_data = list(self._nic_tx)
        rx_data = list(self._nic_rx)
        times = list(self._nic_time)

        if not times:
            return

        ax.fill_between(times, tx_data, alpha=0.3, color=COLORS["cyan"], label="TX")
        ax.plot(times, tx_data, color=COLORS["cyan"], linewidth=1.5)
        ax.fill_between(times, rx_data, alpha=0.3, color=COLORS["purple"], label="RX")
        ax.plot(times, rx_data, color=COLORS["purple"], linewidth=1.5)

        # Auto-scale
        if times:
            xmin = max(0, times[-1] - 120)
            xmax = times[-1] + 5
            ax.set_xlim(xmin, xmax)
        max_val = max(max(tx_data, default=1), max(rx_data, default=1), 1)
        ax.set_ylim(0, max_val * 1.3)

        ax.legend(loc="upper left", fontsize=7, framealpha=0.3,
                 facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                 labelcolor=COLORS["text"])

    # ------------------------------------------------------------------
    def _update_status_panel(self):
        ax = self._axes["status"]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("STATUS", fontsize=11, fontweight="bold",
                     color=COLORS["text"], pad=8)

        y = 0.88
        gap = 0.13

        # Cycle
        ax.text(0.1, y, "Cycle:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        ax.text(0.9, y, str(self._cycle), fontsize=9, color=COLORS["text"],
                ha="right", fontweight="bold")
        y -= gap

        # Elapsed
        ax.text(0.1, y, "Elapsed:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        m, s = divmod(int(self._elapsed), 60)
        ax.text(0.9, y, f"{m}m {s}s", fontsize=9, color=COLORS["text"],
                ha="right")
        y -= gap

        # Model status
        ax.text(0.1, y, "Model:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        model_color = COLORS["green"] if "trained" in self._model_status else COLORS["yellow"]
        ax.text(0.9, y, self._model_status, fontsize=9, color=model_color,
                ha="right", fontweight="bold")
        y -= gap

        # Congestion
        ax.text(0.1, y, "Congestion:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        if self._congested:
            cong_text = f"{len(self._congested)} link(s)"
            ax.text(0.9, y, cong_text, fontsize=9, color=COLORS["red"],
                    ha="right", fontweight="bold")
        else:
            ax.text(0.9, y, "None", fontsize=9, color=COLORS["green"],
                    ha="right")
        y -= gap

        # Reroute
        ax.text(0.1, y, "Reroute:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        if self._reroute_active:
            ax.text(0.9, y, "ACTIVE", fontsize=9, color=COLORS["red"],
                    ha="right", fontweight="bold")
            y -= gap * 0.8
            ax.text(0.1, y, self._reroute_path, fontsize=7,
                    color=COLORS["yellow"], style="italic")
        else:
            ax.text(0.9, y, "Idle", fontsize=9, color=COLORS["green"],
                    ha="right")
        y -= gap

        # NIC info
        ax.text(0.1, y, "NICs:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        ax.text(0.9, y, str(len(self._nic_rates)), fontsize=9,
                color=COLORS["text"], ha="right")

    # ------------------------------------------------------------------
    def _update_timing_panel(self):
        ax = self._axes["timing"]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("ML METRICS", fontsize=11, fontweight="bold",
                     color=COLORS["text"], pad=8)

        y = 0.88
        gap = 0.13

        # Train time
        ax.text(0.1, y, "Train:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        if self._train_time_ms > 0:
            ax.text(0.9, y, f"{self._train_time_ms:.0f} ms",
                    fontsize=9, color=COLORS["cyan"], ha="right")
        else:
            ax.text(0.9, y, "—", fontsize=9, color=COLORS["text_dim"],
                    ha="right")
        y -= gap

        # Predict time
        ax.text(0.1, y, "Predict:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        if self._predict_time_ms > 0:
            ax.text(0.9, y, f"{self._predict_time_ms:.1f} ms",
                    fontsize=9, color=COLORS["cyan"], ha="right")
        else:
            ax.text(0.9, y, "—", fontsize=9, color=COLORS["text_dim"],
                    ha="right")
        y -= gap

        # R² score
        ax.text(0.1, y, "R²:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        if self._r2_score is not None:
            r2_color = COLORS["green"] if self._r2_score > 0.7 else COLORS["yellow"]
            ax.text(0.9, y, f"{self._r2_score:.4f}",
                    fontsize=9, color=r2_color, ha="right", fontweight="bold")
        else:
            ax.text(0.9, y, "—", fontsize=9, color=COLORS["text_dim"],
                    ha="right")
        y -= gap

        # MAE
        ax.text(0.1, y, "MAE:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        if self._mae_score is not None:
            ax.text(0.9, y, f"{self._mae_score:.6f}",
                    fontsize=9, color=COLORS["cyan"], ha="right")
        else:
            ax.text(0.9, y, "—", fontsize=9, color=COLORS["text_dim"],
                    ha="right")
        y -= gap

        # Poll interval
        from utils import POLL_INTERVAL
        ax.text(0.1, y, "Poll Δ:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        ax.text(0.9, y, f"{POLL_INTERVAL}s", fontsize=9,
                color=COLORS["text"], ha="right")
        y -= gap

        # Threshold
        ax.text(0.1, y, "Threshold:", fontsize=9, color=COLORS["text_dim"],
                fontweight="bold")
        ax.text(0.9, y, f"{UTILIZATION_THRESHOLD:.0%}", fontsize=9,
                color=COLORS["yellow"], ha="right")

    # ------------------------------------------------------------------
    def _update_events_panel(self):
        ax = self._axes["events"]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("EVENT LOG", fontsize=11, fontweight="bold",
                     color=COLORS["text"], pad=8)

        events = list(self._events)
        if not events:
            ax.text(0.5, 0.5, "No events yet...", fontsize=9,
                    color=COLORS["text_dim"], ha="center", va="center",
                    style="italic")
            return

        y = 0.92
        for i, event in enumerate(events[:10]):
            alpha = max(0.3, 1.0 - i * 0.07)
            color = COLORS["text"]
            if "CONGESTION" in event.upper() or "⚠" in event:
                color = COLORS["red"]
            elif "REROUTE" in event.upper() or "reroute" in event:
                color = COLORS["yellow"]
            elif "CLEARED" in event.upper() or "✓" in event:
                color = COLORS["green"]
            elif "TRAIN" in event.upper() or "MODEL" in event.upper():
                color = COLORS["cyan"]

            ax.text(0.02, y, event, fontsize=7.5, color=color,
                    alpha=alpha, fontfamily="Consolas",
                    verticalalignment="top")
            y -= 0.095

    # ------------------------------------------------------------------
    def _update_topo_panel(self):
        """Draw mini gauge bars for each key link."""
        ax = self._axes["topo"]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("LINK GAUGES", fontsize=11, fontweight="bold",
                     color=COLORS["text"], pad=8)

        key_links = [
            (("s1", 2), "s1→s2"),
            (("s1", 3), "s1→s3"),
            (("s1", 4), "s1→s4"),
            (("s2", 3), "s2→s3"),
        ]

        y = 0.85
        bar_height = 0.09
        gap = 0.18

        for (sw, port), label in key_links:
            data = self._util_history.get((sw, port), [])
            val = data[-1] if data else 0.0

            # Label
            ax.text(0.05, y + 0.02, label, fontsize=8,
                    color=COLORS["text"], fontweight="bold")
            ax.text(0.92, y + 0.02, f"{val:.0%}", fontsize=8,
                    color=COLORS["text"], ha="right", fontweight="bold")

            # Background bar
            bg_rect = mpatches.FancyBboxPatch(
                (0.05, y - bar_height), 0.87, bar_height,
                boxstyle="round,pad=0.01",
                facecolor=COLORS["grid"], edgecolor=COLORS["border"],
                linewidth=0.5,
            )
            ax.add_patch(bg_rect)

            # Filled bar
            if val > 0.001:
                if val >= UTILIZATION_THRESHOLD:
                    bar_color = COLORS["red"]
                elif val >= 0.5:
                    bar_color = COLORS["yellow"]
                else:
                    bar_color = COLORS["green"]

                fill_width = min(val, 1.0) * 0.87
                fill_rect = mpatches.FancyBboxPatch(
                    (0.05, y - bar_height), fill_width, bar_height,
                    boxstyle="round,pad=0.01",
                    facecolor=bar_color, edgecolor="none",
                    alpha=0.8,
                )
                ax.add_patch(fill_rect)

            y -= gap

    # ==================================================================
    #  RUN (starts the GUI event loop)
    # ==================================================================

    def run(self, interval_ms=1000):
        """
        Start the matplotlib animation loop.
        This BLOCKS the calling thread (run in main thread or a dedicated one).

        Args:
            interval_ms: milliseconds between frame updates
        """
        self._setup_figure()
        self._ani = FuncAnimation(
            self.fig, self._update,
            interval=interval_ms,
            cache_frame_data=False,
        )
        plt.show()

    def close(self):
        """Close the dashboard window."""
        if self.fig:
            plt.close(self.fig)
