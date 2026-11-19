"""
Holonomic Mecanum Offline Dataset Generator for Mobile Robot Docking
=====================================================================
Generates a 6-column supervisory training dataset for offline
behavioral cloning of a 3-output holonomic controller (vx, vy, w) on the
AgileX LIMO mobile robot in Mecanum mode.

Key Improvements:
-----------------
1. Restorative Deceleration Inside 8 cm: The linear and angular braking ramps
   converge all the way to the inner core (d_stop_internal = 1.2 cm, angle = 1.1 deg)
   rather than vanishing at the 8 cm test boundary.
2. Continuous Turning Law at +/-180 deg: Smooth CCW symmetry breaking near +/-pi
   eliminating sign-flipping chattering.
3. Stratified Distance-Band Sampling: Balances near-dock, braking ring, and far-field
   samples across all 360 degrees.
4. Centralized Configuration: Reads defaults from `config.json`.

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
from typing import List, Dict, Tuple, Optional

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.config import load_config

_CFG = load_config()
_KIN = _CFG["kinematics"]
_EXP = _CFG["expert_and_dataset"]

# ==============================================================================
# KINEMATIC AND ARENA PARAMETERS (Loaded from config.json)
# ==============================================================================
V_MAX = float(_KIN["v_max"])
W_MAX = float(_KIN["w_max"])
X_MAX = float(_KIN["x_max"])
Y_MAX = float(_KIN["y_max"])
STEP_XY = float(_EXP["step_xy"])
NUM_ANGLES = int(_EXP["num_angles"])

D_STOP = float(_EXP["d_stop_outer"])
D_STOP_INTERNAL = float(_EXP["d_stop_internal"])
D_STOP_ANGLE_TOL = float(_EXP["d_stop_angle_internal"])
D_BRAKE = float(_EXP["d_brake"])
ANGLE_BRAKE = float(_EXP["angle_brake"])

# Expert supervisor thresholds (config.json -> expert_and_dataset)
RAMP_MIN = float(_EXP["ramp_min"])                # minimum fraction of v_max / w_max on the braking ramps
SYM_BREAK_THETA = float(_EXP["sym_break_theta"])  # e_theta above this -> consistent CCW alignment (rad)
CLASS_V_EPS = float(_EXP["class_v_eps"])          # |v| <= this counts as "no translation" (m/s)
CLASS_W_EPS = float(_EXP["class_w_eps"])          # |w| <= this counts as "no rotation" (rad/s)

SAMPLES_PER_CLASS = int(_EXP["samples_per_class"])
COLUMNS = [
    "x_target_robot",
    "y_target_robot",
    "theta_robot",
    "vx_robot",
    "vy_robot",
    "w_robot",
    "group_id",
]

MECANUM_CLASS_NAMES = {
    1: "Class 1: Terminal Docking Stop (v = 0, w = 0)",
    2: "Class 2: Pure In-Place CCW Rotation (v = 0, w > 0)",
    3: "Class 3: Pure In-Place CW Rotation (v = 0, w < 0)",
    4: "Class 4: Pure Holonomic Translation (v > 0, w = 0)",
    5: "Class 5: Holonomic Translation + CCW Rotation (v > 0, w > 0)",
    6: "Class 6: Holonomic Translation + CW Rotation (v > 0, w < 0)",
}


JITTERED_COUNTS: Dict[int, int] = {}


def recast_angle(angle_rad: float) -> float:
    """Wraps an angle into (-pi, +pi]."""
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def classify_mecanum_maneuver(vx: float, vy: float, w: float) -> int:
    """Classifies a holonomic command (vx, vy, w) into 6 balanced regimes."""
    V_EPS = CLASS_V_EPS
    W_EPS = CLASS_W_EPS
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
    d_stop: float = D_STOP_INTERNAL,
    d_stop_angle_tol: float = D_STOP_ANGLE_TOL,
    d_brake: float = D_BRAKE,
    angle_brake: float = ANGLE_BRAKE,
) -> Tuple[float, float, float, int]:
    """
    Deterministic holonomic expert supervisor for AgileX LIMO Mecanum mode.
    Maintains restorative translational and rotational feedback inside the dock
    zone down to `d_stop_internal` and `d_stop_angle_internal`,
    with vector norm sqrt(vx^2 + vy^2) <= v_max.
    """
    dist = math.hypot(x_tr, y_tr)
    e_theta = recast_angle(theta_r)

    # 1. Translational command (vx, vy) with restorative ramp down to d_stop (1.2 cm)
    if dist <= d_stop:
        vx, vy = 0.0, 0.0
    else:
        if dist < d_brake:
            ramp = min(1.0, max(RAMP_MIN, dist / d_brake))
            v_mag = v_max * ramp
        else:
            v_mag = v_max
        vx = v_mag * (x_tr / dist)
        vy = v_mag * (y_tr / dist)

    # 2. Angular command (w) with restorative alignment down to 0.02 rad and smooth +/-pi handling
    if abs(e_theta) <= d_stop_angle_tol:
        w = 0.0
    elif e_theta > SYM_BREAK_THETA:
        w = w_max
    else:
        if abs(e_theta) < angle_brake:
            ramp_w = min(1.0, max(RAMP_MIN, abs(e_theta) / angle_brake))
        else:
            ramp_w = 1.0
        w = (-w_max if e_theta > 0.0 else w_max) * ramp_w

    m_id = classify_mecanum_maneuver(vx, vy, w)
    return vx, vy, w, m_id


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
    banded_pools: Dict[int, Dict[int, Dict[Tuple[float, float, float], List[float]]]] = {
        m_id: {0: {}, 1: {}, 2: {}} for m_id in range(1, 7)
    }

    def _add_sample(x: float, y: float, th: float) -> None:
        th_w = recast_angle(th)
        vx, vy, w, m_id = expert_mecanum_controller(x, y, th_w)
        dist = math.hypot(x, y)
        band = 0 if dist <= D_STOP else (1 if dist <= 0.80 else 2)  # 0.80 m: sampling band limit
        key = (round(x, 4), round(y, 4), round(th_w, 4))
        if key not in banded_pools[m_id][band]:
            banded_pools[m_id][band][key] = [key[0], key[1], key[2], round(vx, 4), round(vy, 4), round(w, 4)]

        if abs(abs(th_w) - math.pi) > 1e-3:
            my = -y
            mth = recast_angle(-th_w)
            mvx, mvy, mw, mm_id = expert_mecanum_controller(x, my, mth)
            mkey = (round(x, 4), round(my, 4), round(mth, 4))
            if mkey not in banded_pools[mm_id][band]:
                banded_pools[mm_id][band][mkey] = [mkey[0], mkey[1], mkey[2], round(mvx, 4), round(mvy, 4), round(mw, 4)]

    # 1. Primary 5x5m spatial grid (includes th=0 and fine aligned headings for Class 4)
    n_steps_x = int(round(2.0 * x_max / step_xy))
    n_steps_y = int(round(2.0 * y_max / step_xy))
    xs = [round(-x_max + i * step_xy, 2) for i in range(n_steps_x + 1)]
    ys = [round(-y_max + i * step_xy, 2) for i in range(n_steps_y + 1)]
    thetas = [round(-math.pi + i * (2.0 * math.pi / num_angles), 4) for i in range(num_angles)] + [
        round(-D_STOP_ANGLE_TOL + i * (2.0 * D_STOP_ANGLE_TOL / 6), 4) for i in range(7)
    ]
    for x in xs:
        for y in ys:
            for th in thetas:
                _add_sample(x, y, th)

    # 2. High-density terminal docking core (|x|, |y| <= 0.08m, 4mm step, 48 angles)
    dock_step = 0.0040
    n_dock = int(round(0.16 / dock_step))
    dock_xs = [round(-0.08 + i * dock_step, 4) for i in range(n_dock + 1)]
    dock_ys = [round(-0.08 + i * dock_step, 4) for i in range(n_dock + 1)]
    dock_thetas = [round(-math.pi + i * (2.0 * math.pi / 48), 4) for i in range(48)]
    for x in dock_xs:
        for y in dock_ys:
            for th in dock_thetas:
                _add_sample(x, y, th)

    # 2b. Ultra-fine inner stop core (|x|, |y| <= D_STOP_INTERNAL, |th| <= D_STOP_ANGLE_TOL)
    core_xs = [round(-D_STOP_INTERNAL + i * (2.0 * D_STOP_INTERNAL / 16), 4) for i in range(17)]
    core_ys = [round(-D_STOP_INTERNAL + i * (2.0 * D_STOP_INTERNAL / 16), 4) for i in range(17)]
    core_ths = [round(-D_STOP_ANGLE_TOL + i * (2.0 * D_STOP_ANGLE_TOL / 20), 4) for i in range(21)]
    for x in core_xs:
        for y in core_ys:
            for th in core_ths:
                _add_sample(x, y, th)

    # 3. Medium-density braking & approach ring (|x|, |y| <= 0.80m, 4cm step, 32 angles)
    ring_xs = [round(-0.80 + i * 0.04, 3) for i in range(41)]
    ring_ys = [round(-0.80 + i * 0.04, 3) for i in range(41)]
    ring_thetas = [round(-math.pi + (i + 0.5) * (2.0 * math.pi / 32), 4) for i in range(32)]
    for x in ring_xs:
        for y in ring_ys:
            for th in ring_thetas:
                _add_sample(x, y, th)

    # 4. Stratified balancing across distance bands
    balanced: List[List[float]] = []
    final_counts: Dict[int, int] = {}

    for m_id in range(1, 7):
        rng = random.Random(42 + m_id)
        band_lists = [[row + [tuple(row[:3])] for row in banded_pools[m_id][b].values()] for b in (0, 1, 2)]
        non_empty_bands = [b_list for b_list in band_lists if len(b_list) > 0]

        selected: List[List[float]] = []
        if non_empty_bands:
            per_band_quota = samples_per_class // len(non_empty_bands)
            leftover_pool: List[List[float]] = []
            for b_list in non_empty_bands:
                rng.shuffle(b_list)
                take = min(len(b_list), per_band_quota)
                selected.extend(b_list[:take])
                leftover_pool.extend(b_list[take:])

            if len(selected) < samples_per_class and leftover_pool:
                rng.shuffle(leftover_pool)
                need = samples_per_class - len(selected)
                selected.extend(leftover_pool[:need])

            all_unique = [item for b_list in non_empty_bands for item in b_list]
            while len(selected) < samples_per_class:
                base = rng.choice(all_unique)
                jx = round(base[0] + rng.gauss(0.0, 0.0005), 4)
                jy = round(base[1] + rng.gauss(0.0, 0.0005), 4)
                jth = round(recast_angle(base[2] + rng.gauss(0.0, 0.0005)), 4)
                selected.append([jx, jy, jth, base[3], base[4], base[5], base[6]])  # jittered copy inherits the base group

        n_unique_total = sum(len(b) for b in non_empty_bands) if non_empty_bands else 0
        n_jitter = max(0, min(len(selected), samples_per_class) - n_unique_total)
        JITTERED_COUNTS[m_id] = n_jitter
        balanced.extend(selected[:samples_per_class])
        final_counts[m_id] = len(selected[:samples_per_class])

    random.Random(42).shuffle(balanced)
    # group_id: one id per unique state; jittered copies share the id of their base
    # state so that train/validation splits can keep a whole group on one side.
    _gid: Dict[Tuple[float, float, float], int] = {}
    for row in balanced:
        row[6] = _gid.setdefault(row[6], len(_gid))
    return balanced, final_counts


def build_mecanum_dataset(
    output_path: Optional[str] = None,
    samples_per_class: Optional[int] = None,
    config_path: Optional[str] = None,
) -> int:
    """Generates and writes the 6-column Mecanum dataset CSV."""
    cfg = load_config(config_path)
    exp_cfg = cfg["expert_and_dataset"]
    out_csv = output_path or exp_cfg["dataset_csv"]
    spc = int(samples_per_class if samples_per_class is not None else exp_cfg["samples_per_class"])

    resolved = out_csv if os.path.isabs(out_csv) else os.path.join(_project_root, out_csv)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)

    print("=" * 78)
    print("   LIMO MECANUM HOLONOMIC OFFLINE DATASET GENERATOR (3 OUTPUTS)")
    print("=" * 78)
    samples, counts = generate_mecanum_dataset(samples_per_class=spc)

    with open(resolved, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(samples)

    print(f"Saved {len(samples)} samples to: {resolved}")
    print("Class distribution:")
    for m_id in range(1, 7):
        print(f"  {MECANUM_CLASS_NAMES[m_id]:58s} -> {counts[m_id]} | jittered copies: {JITTERED_COUNTS.get(m_id, 0)}")
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3-output Mecanum holonomic dataset.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    args = parser.parse_args()
    build_mecanum_dataset(output_path=args.output, samples_per_class=args.samples_per_class)


if __name__ == "__main__":
    main()
