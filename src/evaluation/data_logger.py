"""
Telemetry and Metrics Logger (DataLogger)
=========================================
Records trajectory, errors, velocity commands, and loss criteria
at each 20 Hz iteration. Fully compliant with ISO 18646-2 metrology processing.

Project: LIMO Offline Learning (CERV / ENIB)
"""

import os
import sys
import csv
import time
import math
from typing import List, Dict, Any, Optional


LIMO_HALF_TRACK = 0.086  # Characteristic half-track: 86 mm


class DataLogger:
    """
    Real-time telemetry and state logger for mobile robot navigation.
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._rows: List[Dict[str, Any]] = []

    def log(
        self,
        position: List[float],
        target: List[float],
        command: List[float],
        grad: Optional[List[float]] = None,
        cost: float = 0.0,
        sim_time: Optional[float] = None
    ) -> None:
        """
        Logs a telemetry sample at the current time step.

        Args:
            position (List[float]): Current pose [x, y, theta].
            target (List[float]): Target pose [tx, ty, ttheta].
            command (List[float]): Applied velocity command [v_lin, v_ang].
            grad (List[float], optional): Gradient [grad_0, grad_1] (default [0.0, 0.0]).
            cost (float): Scalar cost value at the current step.
            sim_time (Optional[float]): Explicit simulation time in seconds (when running faster than real-time).
        """
        if self._start_time is None:
            self._start_time = time.time()

        if grad is None:
            grad = [0.0, 0.0]

        t = float(sim_time) if sim_time is not None else (time.time() - self._start_time)
        x, y, theta = position
        tx, ty, ttheta = target
        v_lin, v_ang = command[0], command[1]

        # Angular error normalized into (-pi, pi]
        e_theta_raw = (theta - ttheta) % (2.0 * math.pi)
        if e_theta_raw > math.pi:
            e_theta_raw -= 2.0 * math.pi

        self._rows.append({
            "timestamp": round(t, 4),
            "x": round(x, 4),
            "y": round(y, 4),
            "theta": round(theta, 4),
            "e_x": round(x - tx, 4),
            "e_y": round(y - ty, 4),
            "e_theta": round(e_theta_raw, 4),
            "v_lin": round(v_lin, 4),
            "v_ang": round(v_ang, 4),
            "v_wheel_left": round(v_lin - v_ang * LIMO_HALF_TRACK, 4),
            "v_wheel_right": round(v_lin + v_ang * LIMO_HALF_TRACK, 4),
            "grad_0": round(grad[0], 6),
            "grad_1": round(grad[1], 6),
            "cost": round(cost, 6),
        })

    def save(self, filename: str) -> None:
        """
        Exports recorded telemetry to a CSV file.

        Args:
            filename (str): Destination CSV filepath.
        """
        if not self._rows:
            print("[DataLogger] No telemetry data to save.")
            return

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)

        print(f"[DataLogger] Telemetry successfully saved to: {filename} ({len(self._rows)} samples)")

    def save_plot(self, png_path: str, show: bool = False) -> None:
        """Generates a 3x3 session diagnostic panel identical to APP-Limo-ros2."""
        if not self._rows:
            return
        try:
            import sys
            import matplotlib
            if "matplotlib.pyplot" not in sys.modules:
                matplotlib.use("Qt5Agg" if show else "Agg")
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except Exception:
            print("[DataLogger] matplotlib is not available — skipping plot generation.")
            return

        t = [r["timestamp"] for r in self._rows]
        x = [r["x"] for r in self._rows]
        y = [r["y"] for r in self._rows]
        th = [r["theta"] for r in self._rows]
        ex = [r["e_x"] for r in self._rows]
        ey = [r["e_y"] for r in self._rows]
        et = [r["e_theta"] for r in self._rows]
        vl = [r["v_lin"] for r in self._rows]
        va = [r["v_ang"] for r in self._rows]
        wl = [r["v_wheel_left"] for r in self._rows]
        wr = [r["v_wheel_right"] for r in self._rows]
        g0 = [r["grad_0"] for r in self._rows]
        g1 = [r["grad_1"] for r in self._rows]
        co = [r["cost"] for r in self._rows]

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(os.path.basename(png_path), fontsize=11)
        gs = GridSpec(3, 3, figure=fig)

        ax = fig.add_subplot(gs[0:2, 0])
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.25, linewidth=1.0)
        ax.axvline(x=0, color="k", linestyle="-", alpha=0.25, linewidth=1.0)
        ax.plot(x, y, "b-", linewidth=1.8, label="Trajectoire", zorder=3)

        # Dynamic heading arrows scaled to arena bounding box
        span_x = max(max(x + [0.0]) - min(x + [0.0]), 1.5)
        span_y = max(max(y + [0.0]) - min(y + [0.0]), 1.5)
        arr_len = max(0.35, max(span_x, span_y) * 0.13)

        th0_deg_signed = math.degrees(th[0])
        th0_deg_360 = (th0_deg_signed + 360.0) % 360.0
        thf_deg_signed = math.degrees(th[-1])

        ax.plot(x[0], y[0], "o", color="#27ae60", markeredgecolor="white", markeredgewidth=1.2,
                markersize=9, label=f"Départ ({x[0]:.1f}, {y[0]:.1f}) | Flèche Verte : θ₀={th0_deg_360:.0f}°", zorder=7)
        ax.plot(x[-1], y[-1], "s", color="#e74c3c", markeredgecolor="white", markeredgewidth=1.2,
                markersize=8, label=f"Arrivée ({x[-1]:.2f}, {y[-1]:.2f}) | Flèche Rouge : θf={thf_deg_signed:+.1f}°", zorder=8)
        ax.plot(0, 0, "r*", markersize=15, label="Cible (0, 0) | Flèche Orange : θ*=0°", zorder=6)

        try:
            import matplotlib.patheffects as pe
            arrow_halo = [pe.withStroke(linewidth=4.5, foreground="white")]
        except Exception:
            arrow_halo = []

        # Intermediate flow arrows along trajectory
        if len(x) > 10:
            cum_d = [0.0]
            for i_pt in range(1, len(x)):
                cum_d.append(cum_d[-1] + math.hypot(x[i_pt] - x[i_pt - 1], y[i_pt] - y[i_pt - 1]))
            total_d = cum_d[-1]
            if total_d > 0.5:
                for frac in (0.35, 0.70):
                    target_d = total_d * frac
                    idx_m = min(range(len(cum_d)), key=lambda k: abs(cum_d[k] - target_d))
                    if 1 <= idx_m < len(x) - 2:
                        dx_m = x[idx_m + 1] - x[idx_m - 1]
                        dy_m = y[idx_m + 1] - y[idx_m - 1]
                        if math.hypot(dx_m, dy_m) > 1e-3:
                            ax.annotate(
                                "",
                                xy=(x[idx_m] + 0.01 * dx_m, y[idx_m] + 0.01 * dy_m),
                                xytext=(x[idx_m] - 0.01 * dx_m, y[idx_m] - 0.01 * dy_m),
                                arrowprops=dict(arrowstyle="-|>", color="#2980b9", lw=1.8, mutation_scale=13),
                                zorder=5
                            )

        # Target desired heading arrow (theta=0)
        xt_tip = arr_len * 1.15
        yt_tip = 0.0
        ann_t = ax.annotate(
            "", xy=(xt_tip, yt_tip), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color="#f39c12", lw=2.4, ls="--", mutation_scale=16),
            zorder=7
        )
        if arrow_halo:
            ann_t.arrow_patch.set_path_effects(arrow_halo)

        # Initial heading arrow (Start direction)
        x0_tip = x[0] + arr_len * math.cos(th[0])
        y0_tip = y[0] + arr_len * math.sin(th[0])
        ann_0 = ax.annotate(
            "", xy=(x0_tip, y0_tip), xytext=(x[0], y[0]),
            arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=3.0, mutation_scale=20),
            zorder=9
        )
        if arrow_halo:
            ann_0.arrow_patch.set_path_effects(arrow_halo)

        # Final heading arrow (Finish direction)
        xf_tip = x[-1] + arr_len * math.cos(th[-1])
        yf_tip = y[-1] + arr_len * math.sin(th[-1])
        ann_f = ax.annotate(
            "", xy=(xf_tip, yf_tip), xytext=(x[-1], y[-1]),
            arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=3.0, mutation_scale=20),
            zorder=10
        )
        if arrow_halo:
            ann_f.arrow_patch.set_path_effects(arrow_halo)

        # Explicit axis limits encompassing all arrow tips so arrows are never clipped
        all_px = x + [0.0, x0_tip, xf_tip, xt_tip]
        all_py = y + [0.0, y0_tip, yf_tip, yt_tip]
        pad_xy = max(0.45, arr_len * 0.75)
        ax.set_xlim(min(all_px) - pad_xy, max(all_px) + pad_xy)
        ax.set_ylim(min(all_py) - pad_xy, max(all_py) + pad_xy)

        ax.set_title("Trajectoire XY & Orientations (Flèches Départ / Arrivée)", fontweight="bold")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="best", fontsize=8.0, framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal", adjustable="box")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, co, "r-", linewidth=1)
        ax2.set_title("Coût")
        ax2.set_xlabel("t (s)")
        ax2.grid(True)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t, ex, label="e_x")
        ax3.plot(t, ey, label="e_y")
        ax3.plot(t, et, label="e_θ")
        ax3.set_title("Erreurs")
        ax3.set_xlabel("t (s)")
        ax3.legend()
        ax3.grid(True)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(t, vl, label="v_lin")
        ax4.plot(t, va, label="v_ang")
        ax4.set_title("Vitesses robot")
        ax4.set_xlabel("t (s)")
        ax4.legend()
        ax4.grid(True)

        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(t, wl, label="gauche")
        ax5.plot(t, wr, label="droite")
        ax5.set_title("Vitesses roues (m/s)")
        ax5.set_xlabel("t (s)")
        ax5.legend()
        ax5.grid(True)

        ax6 = fig.add_subplot(gs[2, 1])
        dists = [math.hypot(xi, yi) for xi, yi in zip(x, y)]
        ax6.plot(t, dists, color="#8e44ad", linewidth=1.5, label="distance (m)")
        ax6.axhline(y=0.40, color="#e67e22", linestyle="--", label="freinage (0.40m)")
        ax6.axhline(y=0.08, color="#27ae60", linestyle="-.", label="arrêt cible (0.08m)")
        ax6.set_title("Distance à la cible (m)")
        ax6.set_xlabel("t (s)")
        ax6.legend(fontsize=8)
        ax6.grid(True)

        ax7 = fig.add_subplot(gs[2, 2])
        ax7.plot(t, x, label="x")
        ax7.plot(t, y, label="y")
        ax7.plot(t, th, label="θ")
        ax7.set_title("Position")
        ax7.set_xlabel("t (s)")
        ax7.legend()
        ax7.grid(True)

        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"[DataLogger] Session plot saved to: {png_path}")

    def reset(self) -> None:
        """Resets telemetry accumulators."""
        self._start_time = None
        self._rows = []
