"""
Centralized Configuration & Unified ISO 18646-2 Docking Criteria (Mecanum)
==========================================================================
Loads project parameters from `config.json` at the repository root (the single
source of truth: no default values are hardcoded in the code) and provides the
shared criteria for docking arrival, velocity/motion
stationarity, and dwell-time verification across all execution, benchmarking,
visualization, and metrological evaluation tools.
"""

from __future__ import annotations

import os
import json
import math
import copy
from typing import Dict, Any, Optional, Sequence, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")


class ConfigSection(dict):
    """
    Dictionary view of one `config.json` block. There are NO hidden default
    values: reading a key that is absent from `config.json` (with `[]` or
    `.get()`) raises a KeyError naming the missing key, so `config.json`
    remains the single source of truth for every main parameter.
    """

    def __init__(self, data: Dict[str, Any], path: str, source: str):
        super().__init__()
        self._path = path
        self._source = source
        for k, v in data.items():
            child = f"{path}.{k}" if path else k
            self[k] = ConfigSection(v, child, source) if isinstance(v, dict) else v

    def __missing__(self, key: str) -> Any:
        full = f"{self._path}.{key}" if self._path else key
        raise KeyError(f"{self._source}: missing required key '{full}'")

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self[key]


_CACHE: Dict[str, ConfigSection] = {}


def load_config(config_path: Optional[str] = None) -> ConfigSection:
    """
    Loads the project parameters from `config.json` (repository root), or from
    the file given by the `LIMO_CONFIG_PATH` environment variable.

    The file is mandatory and is the only source of parameters: a missing file,
    an invalid JSON or a missing key stops the program with an explicit error.
    """
    path = config_path or os.environ.get("LIMO_CONFIG_PATH") or _DEFAULT_CONFIG_PATH
    if not os.path.isabs(path):
        candidate = os.path.join(_PROJECT_ROOT, path)
        if not os.path.exists(path) and os.path.exists(candidate):
            path = candidate
    path = os.path.abspath(path)

    if path not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top-level JSON value must be an object")
        _CACHE[path] = ConfigSection(data, "", os.path.basename(path))
    return copy.deepcopy(_CACHE[path])


def effective_speed_limits(cfg: Dict[str, Any], speed_scale: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Single rule for the velocity limits applied by every execution mode
    (kinematic simulation, Gazebo and the physical robot).

    The network outputs are scaled as  v = out_v * v_max_eff,  w = out_w * w_max,
    with  v_max_eff = v_max * speed_scale  and speed_scale clamped to [0.2, 1.0].
    `speed_scale` comes from `config.json` (kinematics.speed_scale) unless an
    explicit override (e.g. the --speed-scale command-line option) is given.

    Returns:
        (speed_scale_effective, v_max_effective, w_max)
    """
    kin = cfg["kinematics"]
    scale = float(kin["speed_scale"] if speed_scale is None else speed_scale)
    scale = max(0.2, min(1.0, scale))
    return scale, float(kin["v_max"]) * scale, float(kin["w_max"])


def recast_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def check_docking_and_stationarity(
    dist_xy: float,
    e_theta: float,
    v_cmd: float,
    w_cmd: float,
    pos_window: Sequence[Tuple[float, float, float, float]],
    dist_tol: Optional[float] = None,
    angle_tol: Optional[float] = None,
    vy_cmd: float = 0.0,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, bool]:
    """
    Unified arrival and stationarity check across all online/simulated loops.

    Returns:
        (in_dock_envelope, is_stationary)
    """
    dock_cfg = (cfg or load_config())["docking_and_iso"]
    d_tol = float(dist_tol if dist_tol is not None else dock_cfg["dist_tol"])
    a_tol = float(angle_tol if angle_tol is not None else dock_cfg["angle_tol"])
    vel_tol = float(dock_cfg["stopped_vel_tol"])
    ang_tol = float(dock_cfg["stopped_ang_tol"])
    move_tol = float(dock_cfg["stopped_move_window_m"])
    rot_tol = float(dock_cfg["stopped_rot_window_rad"])
    win_n = int(dock_cfg["window_samples"])

    if not (math.isfinite(dist_xy) and math.isfinite(e_theta) and math.isfinite(v_cmd) and math.isfinite(w_cmd) and math.isfinite(vy_cmd)):
        return False, False

    in_dock_envelope = (dist_xy <= d_tol) and (abs(e_theta) <= a_tol)

    v_mag = math.hypot(v_cmd, vy_cmd)
    is_stopped_cmds = (v_mag <= vel_tol) and (abs(w_cmd) <= ang_tol)

    if len(pos_window) >= win_n:
        _, x_old, y_old, th_old = pos_window[0]
        _, x_now, y_now, th_now = pos_window[-1]
        d_moved = math.hypot(x_now - x_old, y_now - y_old)
        d_rot = abs(recast_angle_rad(th_now - th_old))
        is_stopped_motion = (d_moved <= move_tol) and (d_rot <= rot_tol)
        is_stationary = is_stopped_cmds and is_stopped_motion
    else:
        is_stationary = False

    return in_dock_envelope, is_stationary


def evaluate_csv_trial_convergence(
    rows: Sequence[Dict[str, Any]],
    dist_tol: Optional[float] = None,
    angle_tol: Optional[float] = None,
    dwell_time_required: Optional[float] = None,
    max_duration: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Unified post-hoc CSV trial evaluator (fallback used by `benchmark_visualizer.py`
    and `calculate_iso_metrics.py` when the live verdict is not available).

    A trial is successful only if the FINAL CONTINUOUS segment of the log is
    entirely inside the docking envelope (distance <= dist_tol, |e_theta| <=
    angle_tol) with stopped commands (|v| <= stopped_vel_tol, |w| <= stopped_ang_tol)
    and lasts at least `dwell_time - 2*dt` (the 2*dt margin accounts for the last
    cycle, which is not logged at the moment the live loop declares success),
    and the trial ended before `max_duration`.
    """
    if not rows:
        return False

    the_cfg = cfg or load_config()
    dock_cfg = the_cfg["docking_and_iso"]
    d_tol = float(dist_tol if dist_tol is not None else dock_cfg["dist_tol"])
    a_tol = float(angle_tol if angle_tol is not None else dock_cfg["angle_tol"])
    vel_tol = float(dock_cfg["stopped_vel_tol"])
    ang_tol = float(dock_cfg["stopped_ang_tol"])
    dwell_req = float(dwell_time_required if dwell_time_required is not None else dock_cfg["dwell_time"])
    max_dur = float(max_duration if max_duration is not None else dock_cfg["max_duration"])
    dt = float(the_cfg["kinematics"]["dt"])

    def _parse(row: Dict[str, Any]) -> Optional[Tuple[float, bool]]:
        try:
            t = float(row.get("timestamp", 0.0))
            x = float(row.get("x", 0.0))
            y = float(row.get("y", 0.0))
            th = float(row.get("theta", 0.0))
            ex = float(row.get("e_x", x))
            ey = float(row.get("e_y", y))
            eth = recast_angle_rad(float(row.get("e_theta", th)))
            vl = float(row.get("v_lin", 0.0))
            vy = float(row.get("v_y", 0.0))
            va = float(row.get("v_ang", 0.0))
        except (ValueError, TypeError):
            return None
        if not all(math.isfinite(val) for val in (t, x, y, th, ex, ey, eth, vl, vy, va)):
            return None
        docked = (math.hypot(ex, ey) <= d_tol and abs(eth) <= a_tol
                  and math.hypot(vl, vy) <= vel_tol and abs(va) <= ang_tol)
        return t, docked

    first = _parse(rows[0])
    last = _parse(rows[-1])
    if first is None or last is None or not last[1]:
        return False
    if last[0] - first[0] >= max_dur:
        return False

    # Walk back from the last row while the robot stays docked and stopped.
    seg_start_t = last[0]
    for row in reversed(rows):
        parsed = _parse(row)
        if parsed is None or not parsed[1]:
            break
        seg_start_t = parsed[0]

    return (last[0] - seg_start_t) >= (dwell_req - 2.0 * dt)
