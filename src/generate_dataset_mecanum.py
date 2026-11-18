"""
Holonomic Mecanum Offline Dataset Generator for Mobile Robot Docking
=====================================================================
Generates a separate 6-column supervisory training dataset for offline
behavioral cloning of a 3-output holonomic controller (vx, vy, w) on the
AgileX LIMO mobile robot in Mecanum mode.

Does NOT modify any existing differential-drive files or datasets.

Coordinate System & Schema:
---------------------------
Robot-Centric Body Frame representation (6 columns):
    1. x_target_robot : Relative X coordinate of the dock in robot body frame (m)
    2. y_target_robot : Relative Y coordinate of the dock in robot body frame (m)
    3. theta_robot    : Heading error relative to the dock axis [-pi, +pi] (rad)
    4. vx_robot       : Supervisory longitudinal velocity [-V_MAX, +V_MAX] (m/s)
    5. vy_robot       : Supervisory lateral (strafe) velocity [-V_MAX, +V_MAX] (m/s)
    6. w_robot        : Supervisory yaw angular velocity [-W_MAX, +W_MAX] (rad/s)

Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB)
Standards: ROS REP-103 & ISO 18646-2
"""

from __future__ import annotations

import os
import sys
import csv
import math
import random
import argparse
from typing import List, Dict, Tuple

# Dynamic project root resolution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ==============================================================================
# KINEMATIC AND ARENA PARAMETERS (AgileX LIMO Mecanum Holonomic Kinematics)
# ==============================================================================
V_MAX = 0.50                   # Maximum translational speed sqrt(vx^2 + vy^2) (m/s)
W_MAX = 1.00                   # Maximum yaw angular velocity (rad/s)
X_MAX = 5.0                    # Global arena half-width X [-5.0, +5.0] m
Y_MAX = 5.0                    # Global arena half-length Y [-5.0, +5.0] m
STEP_XY = 0.20                 # Primary spatial regular grid resolution (m)
NUM_ANGLES = 24                # Angular resolution: 24 equidistant headings (15° steps)

D_STOP = 0.080                 # Physical dock capture radius for terminal stop (8.0 cm)
D_STOP_ANGLE_TOL = 0.150       # Heading alignment tolerance for terminal stop (~8.6°)
D_BRAKE = 0.40                 # Linear deceleration funnel distance (40.0 cm)
ANGLE_BRAKE = 0.60             # Angular deceleration funnel threshold (rad)

SAMPLES_PER_CLASS = 7500       # 6 classes * 7,500 = 45,000 balanced samples
COLUMNS = [
    "x_target_robot",
    "y_target_robot",
    "theta_robot",
    "vx_robot",
    "vy_robot",
    "w_robot",
]

MECANUM_CLASS_NAMES = {
    1: "Class 1: Terminal Docking Stop (v = 0, w = 0)",
    2: "Class 2: Pure In-Place CCW Rotation (v = 0, w > 0)",
    3: "Class 3: Pure In-Place CW Rotation (v = 0, w < 0)",
    4: "Class 4: Pure Holonomic Translation (v > 0, w = 0)",
    5: "Class 5: Holonomic Translation + CCW Rotation (v > 0, w > 0)",
    6: "Class 6: Holonomic Translation + CW Rotation (v > 0, w < 0)",
}


def recast_angle(angle_rad: float) -> float:
    """Wraps an angle into [-pi, +pi], preserving exact +/-pi boundaries."""
    if abs(angle_rad - math.pi) < 1e-4:
        return math.pi
    if abs(angle_rad + math.pi) < 1e-4:
        return -math.pi
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def classify_mecanum_maneuver(vx: float, vy: float, w: float) -> int:
    """Classifies a holonomic command (vx, vy, w) into 6 balanced regimes."""
    V_EPS = 0.025
    W_EPS = 0.040
    v_mag = math.hypot(vx, vy)

    if v_mag <= V_EPS:
        if abs(w) <= W_EPS:
            return 1
        return 2 if w > 0.0 else 3

    if abs(w) <= W_EPS:
        return 4
    return 5 if w > 0.0 else 6


def expert_mecanum_controller(
    x_tr: float,
    y_tr: float,
    theta_r: float,
    v_max: float = V_MAX,
    w_max: float = W_MAX,
    d_stop: float = D_STOP,
    d_stop_angle_tol: float = D_STOP_ANGLE_TOL,
    d_brake: float = D_BRAKE,
    angle_brake: float = ANGLE_BRAKE,
) -> Tuple[float, float, float, int]:
    """
    Deterministic holonomic expert supervisor for AgileX LIMO Mecanum mode.
    Decouples translational approach (vx, vy) from yaw alignment (w) with
    smooth linear and angular deceleration funnels.
    """
    dist = math.hypot(x_tr, y_tr)
    e_theta = recast_angle(theta_r)

    # 1. Translational command (vx, vy)
    if dist <= d_stop:
        vx, vy = 0.0, 0.0
    else:
        if dist < d_brake:
            ramp = max(0.0, min(1.0, (dist - d_stop) / (d_brake - d_stop)))
            v_mag = v_max * ramp
        else:
            v_mag = v_max
        vx = v_mag * (x_tr / dist)
        vy = v_mag * (y_tr / dist)

    # 2. Angular command (w)
    if abs(e_theta) <= d_stop_angle_tol:
        w = 0.0
    elif abs(abs(e_theta) - math.pi) < 1e-4:
        # Consistent CCW symmetry breaking at +/-pi
        w = w_max
    else:
        if abs(e_theta) < angle_brake:
            ramp_w = max(0.0, min(1.0, (abs(e_theta) - d_stop_angle_tol) / (angle_brake - d_stop_angle_tol)))
        else:
            ramp_w = 1.0
        w = (-w_max if e_theta > 0.0 else w_max) * ramp_w

    m_id = classify_mecanum_maneuver(vx, vy, w)
    return vx, vy, w, m_id


def _sample_grid(
    xs: List[float],
    ys: List[float],
    thetas: List[float],
    buckets: Dict[int, List[List[float]]],
) -> None:
    """Evaluates the expert controller on a Cartesian-angular grid."""
    for x in xs:
        for y in ys:
            for th in thetas:
                vx, vy, w, m_id = expert_mecanum_controller(x, y, th)
                buckets[m_id].append([x, y, th, vx, vy, w])


def generate_mecanum_dataset(
    x_max: float = X_MAX,
    y_max: float = Y_MAX,
    step_xy: float = STEP_XY,
    num_angles: int = NUM_ANGLES,
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> Tuple[List[List[float]], Dict[int, int]]:
    """
    Constructs a stratified, bilaterally symmetric 3-output holonomic dataset.
    """
    raw_by_class: Dict[int, List[List[float]]] = {i: [] for i in range(1, 7)}

    # 1. Primary 5x5m spatial grid
    n_steps_x = int(round(2.0 * x_max / step_xy))
    n_steps_y = int(round(2.0 * y_max / step_xy))
    xs = [round(-x_max + i * step_xy, 2) for i in range(n_steps_x + 1)]
    ys = [round(-y_max + i * step_xy, 2) for i in range(n_steps_y + 1)]
    thetas = [round(-math.pi + i * (2.0 * math.pi / num_angles), 4) for i in range(num_angles + 1)]
    _sample_grid(xs, ys, thetas, raw_by_class)

    # 2. High-density terminal docking sub-grid (|x|, |y| <= 0.08m, 2mm step, 64 angles)
    dock_step = 0.0020
    n_dock = int(round(0.16 / dock_step))
    dock_xs = [round(-0.08 + i * dock_step, 4) for i in range(n_dock + 1)]
    dock_ys = [round(-0.08 + i * dock_step, 4) for i in range(n_dock + 1)]
    dock_thetas = [round(-math.pi + i * (2.0 * math.pi / 64), 4) for i in range(65)]
    _sample_grid(dock_xs, dock_ys, dock_thetas, raw_by_class)

    # 3. Medium-density braking & approach ring (|x|, |y| <= 0.60m, 2cm step, 32 angles)
    ring_xs = [round(-0.60 + i * 0.02, 3) for i in range(61)]
    ring_ys = [round(-0.60 + i * 0.02, 3) for i in range(61)]
    ring_thetas = [round(-math.pi + i * (2.0 * math.pi / 32), 4) for i in range(33)]
    _sample_grid(ring_xs, ring_ys, ring_thetas, raw_by_class)

    # 4. Bilateral sagittal mirroring: (x, -y, -theta, vx, -vy, -w)
    mirrored_by_class: Dict[int, List[List[float]]] = {i: [] for i in range(1, 7)}
    for m_id in range(1, 7):
        for s in raw_by_class[m_id]:
            if abs(abs(s[2]) - math.pi) < 1e-3:
                continue
            mx, my = s[0], -s[1]
            mth = recast_angle(-s[2])
            mvx, mvy, mw = s[3], -s[4], -s[5]
            new_id = classify_mecanum_maneuver(mvx, mvy, mw)
            mirrored_by_class[new_id].append([mx, my, mth, mvx, mvy, mw])
    for m_id in range(1, 7):
        raw_by_class[m_id].extend(mirrored_by_class[m_id])

    # 5. Periodic boundary sewing at +/-pi
    for m_id in range(1, 7):
        sewn = []
        for s in raw_by_class[m_id]:
            if abs(s[2] - (-math.pi)) < 1e-3:
                sewn.append([s[0], s[1], round(math.pi, 4), s[3], s[4], s[5]])
            elif abs(s[2] - math.pi) < 1e-3:
                sewn.append([s[0], s[1], round(-math.pi, 4), s[3], s[4], s[5]])
        raw_by_class[m_id].extend(sewn)

    # 6. Deduplication by state and stratified balancing
    balanced: List[List[float]] = []
    final_counts: Dict[int, int] = {}
    random.seed(42)

    for m_id in range(1, 7):
        unique_map = {}
        for s in raw_by_class[m_id]:
            key = (round(s[0], 4), round(s[1], 4), round(s[2], 4))
            if key not in unique_map:
                unique_map[key] = [
                    key[0], key[1], key[2],
                    round(s[3], 4), round(s[4], 4), round(s[5], 4),
                ]
        pool = list(unique_map.values())
        random.seed(42 + m_id)
        if len(pool) >= samples_per_class:
            selected = random.sample(pool, samples_per_class)
        else:
            selected = list(pool)
            while len(selected) < samples_per_class:
                base = random.choice(pool)
                jx = round(base[0] + random.gauss(0.0, 0.0005), 4)
                jy = round(base[1] + random.gauss(0.0, 0.0005), 4)
                jth = round(recast_angle(base[2] + random.gauss(0.0, 0.0005)), 4)
                selected.append([jx, jy, jth, base[3], base[4], base[5]])

        balanced.extend(selected)
        final_counts[m_id] = len(selected)

    random.shuffle(balanced)
    return balanced, final_counts


def build_mecanum_dataset(
    output_path: str = "data/dataset_mecanum.csv",
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> int:
    """Generates and writes the 6-column Mecanum dataset CSV."""
    resolved = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)

    print("=" * 78)
    print("   LIMO MECANUM HOLONOMIC OFFLINE DATASET GENERATOR (3 OUTPUTS)")
    print("=" * 78)
    samples, counts = generate_mecanum_dataset(samples_per_class=samples_per_class)

    with open(resolved, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(samples)

    print(f"Saved {len(samples)} samples to: {resolved}")
    print("Class distribution:")
    for m_id in range(1, 7):
        print(f"  {MECANUM_CLASS_NAMES[m_id]:58s} -> {counts[m_id]}")
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3-output Mecanum holonomic dataset.")
    parser.add_argument("--output", type=str, default="data/dataset_mecanum.csv")
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    args = parser.parse_args()
    build_mecanum_dataset(output_path=args.output, samples_per_class=args.samples_per_class)


if __name__ == "__main__":
    main()
