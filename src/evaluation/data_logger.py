"""
Telemetry Logger (DataLogger)
=============================
Records, at every control cycle (20 Hz), the robot pose, the error to the target
and the velocity commands sent by the PioneerNN, then saves them as CSV and as a
compact per-trial figure.

CSV columns:
    timestamp            time since the first sample (s)
    x, y, theta          robot pose given by its odometry (m, m, rad)
    e_x, e_y, e_theta    pose minus target (m, m, rad wrapped to (-pi, pi])
    v_lin, v_ang, v_y    commands: linear (m/s), angular (rad/s), lateral (m/s, Mecanum only)

Project: LIMO Offline Learning (CERV / ENIB)
"""

import os
import csv
import time
import math
from typing import List, Dict, Any, Optional


class DataLogger:
    """Real-time telemetry logger for one docking trial."""

    def __init__(self):
        self._start_time: Optional[float] = None
        self._rows: List[Dict[str, Any]] = []

    def log(
        self,
        position: List[float],
        target: List[float],
        command: List[float],
        sim_time: Optional[float] = None,
    ) -> None:
        """
        Logs one telemetry sample.

        Args:
            position: current pose [x, y, theta].
            target: target pose [tx, ty, ttheta].
            command: applied command [v_lin, v_ang] or [v_lin, v_ang, v_y]
                (v_y = lateral velocity, non-zero only for the holonomic Mecanum robot).
            sim_time: explicit simulation time (s) when running faster than real time.
        """
        if self._start_time is None:
            self._start_time = time.time()

        t = float(sim_time) if sim_time is not None else (time.time() - self._start_time)
        x, y, theta = position
        tx, ty, ttheta = target
        v_lin, v_ang = command[0], command[1]
        v_y = command[2] if len(command) > 2 else 0.0
        e_theta = math.atan2(math.sin(theta - ttheta), math.cos(theta - ttheta))

        self._rows.append({
            "timestamp": round(t, 4),
            "x": round(x, 4),
            "y": round(y, 4),
            "theta": round(theta, 4),
            "e_x": round(x - tx, 4),
            "e_y": round(y - ty, 4),
            "e_theta": round(e_theta, 4),
            "v_lin": round(v_lin, 4),
            "v_ang": round(v_ang, 4),
            "v_y": round(v_y, 4),
        })

    def save(self, filename: str) -> None:
        """Exports the recorded telemetry to a CSV file."""
        if not self._rows:
            print("[DataLogger] No telemetry data to save.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[DataLogger] Telemetry successfully saved to: {filename} ({len(self._rows)} samples)")

    def save_plot(self, png_path: str) -> None:
        """
        Compact per-trial figure: trajectory (left), distance and heading error to
        the target (top right), velocity commands (bottom right).
        """
        if not self._rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            print("[DataLogger] matplotlib is not available — skipping plot generation.")
            return

        r = self._rows
        t = [s["timestamp"] - r[0]["timestamp"] for s in r]
        x = [s["x"] for s in r]
        y = [s["y"] for s in r]
        tx, ty = x[0] - r[0]["e_x"], y[0] - r[0]["e_y"]
        dist = [math.hypot(s["e_x"], s["e_y"]) * 100.0 for s in r]
        eth = [abs(math.degrees(s["e_theta"])) for s in r]

        fig = plt.figure(figsize=(12, 5.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.4])

        ax = fig.add_subplot(gs[:, 0])
        ax.plot(x, y, color="#1f77b4", linewidth=1.6, label="Trajectoire (odométrie)")
        L = 0.15
        for k, color, lab in ((0, "#2ca02c", "Départ"), (-1, "#d62728", "Arrivée")):
            th = r[k]["theta"]
            ax.annotate("", xy=(x[k] + L * math.cos(th), y[k] + L * math.sin(th)), xytext=(x[k], y[k]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
            ax.plot(x[k], y[k], "o", color=color, label=lab)
        ax.plot(tx, ty, "k+", markersize=12, label="Cible")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Trajectoire")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(t, dist, color="#1f77b4", label="Distance à la cible (cm)")
        ax1.set_ylabel("Distance (cm)")
        ax1b = ax1.twinx()
        ax1b.plot(t, eth, color="#d62728", label="|écart de cap| (°)")
        ax1b.set_ylabel("|e_θ| (°)")
        h = ax1.get_legend_handles_labels()[0] + ax1b.get_legend_handles_labels()[0]
        ax1.legend(h, [l.get_label() for l in h], fontsize=8, loc="upper right")
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
        ax2.plot(t, [s["v_lin"] for s in r], color="#2ca02c", label="v (m/s)")
        ax2.plot(t, [s["v_ang"] for s in r], color="#9467bd", label="ω (rad/s)")
        if any(abs(s["v_y"]) > 0 for s in r):
            ax2.plot(t, [s["v_y"] for s in r], color="#ff7f0e", label="v_y (m/s)")
        ax2.set_xlabel("Temps (s)")
        ax2.set_ylabel("Commande")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"[DataLogger] Session plot saved to: {png_path}")

    def reset(self) -> None:
        """Resets telemetry accumulators."""
        self._start_time = None
        self._rows = []
