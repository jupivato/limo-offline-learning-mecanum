"""
Closed-Loop Autonomous Navigation and ISO 18646-2 Metrological Positioning (Mecanum)
====================================================================================
Executes closed-loop autonomous navigation with frozen synaptic weights on the 
AgileX LIMO mobile robot (Mecanum mode) at 20 Hz, on the physical robot only
(there is no official Gazebo model of the Mecanum LIMO).

Key Safety & Metrological Features:
-----------------------------------
1. Centralized Configuration: Reads defaults from `config.json`.
2. Odometry NaN Filtering & Staleness Watchdog: Halts immediately if odometry
   contains non-finite values or freezes beyond `odom_watchdog_timeout_s`.
3. Vectorial Velocity Limiting: Clamps translational speed by Euclidean norm
   sqrt(vx^2 + vy^2) <= V_MAX (preventing 0.71 m/s diagonal overspeed).
4. Known starting pose: --teleport-to X Y THETA recalibrates the odometry on a pose
   marked on the floor (the robot is placed there by hand).
5. Unified ISO 18646-2 Docking & Dwell Verification: Uses `check_docking_and_stationarity`.

Standards Compliance: ROS REP-103 & ISO 18646-2.
Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB).
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import argparse
import collections
from datetime import datetime
from typing import List, Optional, Tuple

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import rclpy
    ROS2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ROS2_AVAILABLE = False

from src.core.config import load_config, check_docking_and_stationarity, effective_speed_limits
from src.core.pioneer_nn import PioneerNN, recast_angle
from src.evaluation.data_logger import DataLogger

try:
    from src.ros2.limo_interface import LimoROS2Interface, sanitize_and_clamp_twist
except (ImportError, ModuleNotFoundError):
    LimoROS2Interface = None
    sanitize_and_clamp_twist = None

_CFG = load_config()
_KIN = _CFG["kinematics"]
_DOCK = _CFG["docking_and_iso"]
_NET = _CFG["network_and_training"]

V_MAX = float(_KIN["v_max"])
W_MAX = float(_KIN["w_max"])
X_MAX = float(_KIN["x_max"])
Y_MAX = float(_KIN["y_max"])
HALF_TRACK_B = float(_KIN["half_track_b"])

DEFAULT_DIST_TOL = float(_DOCK["dist_tol"])
DEFAULT_ANGLE_TOL = float(_DOCK["angle_tol"])
DEFAULT_DWELL_TIME = float(_DOCK["dwell_time"])
DEFAULT_MAX_DURATION = float(_DOCK["max_duration"])


DEFAULT_WEIGHTS_MECANUM = _NET["weights_path"]


def compute_se2_distance(dx: float, dy: float, dtheta: float, b: float = HALF_TRACK_B) -> Tuple[float, float]:
    """
    Computes both Euclidean Cartesian distance and SE(2) composite pose distance.
    """
    d_xy = math.hypot(dx, dy)
    d_pose = math.hypot(d_xy, b * dtheta)
    return d_xy, d_pose


def clamp_mecanum_velocity(
    vx: float,
    vy: float,
    w: float,
    v_max_eff: float = V_MAX,
    w_max_eff: float = W_MAX,
) -> Tuple[float, float, float]:
    """
    Sanitizes (vx, vy, w) against NaN/Inf and enforces Euclidean vector norm
    sqrt(vx^2 + vy^2) <= v_max_eff so diagonal speed never exceeds v_max_eff.
    """
    if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(w)):
        return 0.0, 0.0, 0.0
    v_mag = math.hypot(vx, vy)
    if v_mag > v_max_eff and v_mag > 0.0:
        scale = v_max_eff / v_mag
        vx *= scale
        vy *= scale
    w = max(-w_max_eff, min(w_max_eff, w))
    return vx, vy, w


def run_autonomous_session(
    weights_path: Optional[str] = None,
    target: Optional[List[float]] = None,
    teleport_to: Optional[List[float]] = None,
    is_real: bool = False,
    max_duration: Optional[float] = None,
    dist_tol: Optional[float] = None,
    angle_tol: Optional[float] = None,
    dwell_time_required: Optional[float] = None,
    output_dir: str = "data/evaluation",
    odom_topic: Optional[str] = None,
    cmd_vel_topic: Optional[str] = None,
    speed_scale: Optional[float] = None,
    mode: str = "mecanum",
    config_path: Optional[str] = None,
) -> bool:
    """
    Orchestrates a closed-loop autonomous docking trial with stationary dwell verification.
    """
    if not ROS2_AVAILABLE:
        raise RuntimeError("ROS 2 Humble / Foxy environment is required for live autonomous execution.")

    cfg = load_config(config_path)
    kin_cfg = cfg["kinematics"]
    dock_cfg = cfg["docking_and_iso"]
    net_cfg = cfg["network_and_training"]
    ros_cfg = cfg["ros2_and_safety"]

    if not is_real:
        raise SystemExit(
            "There is no official Gazebo model of the LIMO in Mecanum mode: this repository runs on the "
            "physical robot only. Use --real (see docs/ROBOTS.md)."
        )
    is_mecanum = (mode == "mecanum")
    if weights_path is None:
        if not is_mecanum:
            raise SystemExit("--mode diff requires --weights <differential weights JSON> (they are not part of this repository).")
        weights_path = DEFAULT_WEIGHTS_MECANUM
    output_size = 3 if is_mecanum else 2

    target = target if target is not None else [0.0, 0.0, 0.0]
    max_duration = float(max_duration if max_duration is not None else dock_cfg["max_duration"])
    dist_tol = float(dist_tol if dist_tol is not None else dock_cfg["dist_tol"])
    angle_tol = float(angle_tol if angle_tol is not None else dock_cfg["angle_tol"])
    dwell_time_required = float(dwell_time_required if dwell_time_required is not None else dock_cfg["dwell_time"])
    odom_topic = odom_topic if odom_topic is not None else ros_cfg["odom_topic"]
    cmd_vel_topic = cmd_vel_topic if cmd_vel_topic is not None else ros_cfg["cmd_vel_topic"]
    eff_scale, v_max_effective, w_max_effective = effective_speed_limits(cfg, speed_scale)
    dt = float(kin_cfg["dt"])
    win_n = int(dock_cfg["window_samples"])

    print("\n" + "=" * 78)
    print("   AGILEX LIMO CLOSED-LOOP AUTONOMOUS DOCKING (ISO 18646-2 BENCHMARK)")
    print("=" * 78)
    print(f"Kinematic Mode       : {'Mecanum Holonomic (3 outputs: vx, vy, w)' if is_mecanum else 'Differential (2 outputs: v, w)'}")
    print(f"Weights Artifact     : {weights_path}")
    print(f"Dock Target [x,y,θ]  : x={target[0]:.3f} m, y={target[1]:.3f} m, θ={math.degrees(target[2]):.1f}°")
    print("Platform Mode        : Physical LIMO Hardware")
    print(f"Cruise Limits        : ||V||_max = {v_max_effective:.2f} m/s | W_max = {w_max_effective:.2f} rad/s (Scale: {eff_scale*100:.0f}%)")
    print(f"Capture Threshold    : Distance <= {dist_tol * 100:.1f} cm | Angle <= {math.degrees(angle_tol):.1f}°")
    print(f"Stationary Dwell     : {dwell_time_required:.1f} s continuous immobilization")
    print(f"Mission Duration Max : {max_duration:.1f} s")
    print("=" * 78 + "\n")

    model = PioneerNN(
        input_size=int(net_cfg["input_size"]),
        hidden_size=int(net_cfg["hidden_size"]),
        output_size=output_size,
        use_bias=bool(net_cfg["use_bias"]),
    )
    model.load_from_json_file(weights_path)
    model.eval()
    print(f"[PioneerNN] Policy loaded from: {weights_path} (mode={mode}, output_size={output_size}, use_bias={model.use_bias})")

    rclpy.init()
    robot = LimoROS2Interface(safety_limits=is_real, odom_topic=odom_topic, cmd_vel_topic=cmd_vel_topic)
    logger = DataLogger()

    init_pos = robot.get_position()
    init_dist = math.hypot(init_pos[0] - target[0], init_pos[1] - target[1])
    print(f"[Odometry] Current robot pose: x={init_pos[0]:.3f} m, y={init_pos[1]:.3f} m, theta={math.degrees(init_pos[2]):.1f}° (Distance to target: {init_dist:.2f} m)")

    if teleport_to is not None:
        print(f"[Hardware] Calibrating physical starting pose relative to target: {teleport_to}")
        robot.reset_pose(teleport_to[0], teleport_to[1], teleport_to[2])
        time.sleep(0.2)
    elif init_dist < dist_tol:
        print(f"[Warning] Robot is already at the dock ({init_dist*100:.1f} cm <= {dist_tol*100:.1f} cm).")

    if is_real:
        try:
            input("\n[Ready] Press Enter to start autonomous docking...")
        except (KeyboardInterrupt, EOFError):
            print("\n[Aborted] Mission cancelled by user.")
            robot.cleanup()
            rclpy.shutdown()
            return False

    success = False
    start_time = time.time()
    dwell_timer = 0.0
    dock_arrival_announced = False
    pos_window: collections.deque[Tuple[float, float, float, float]] = collections.deque(maxlen=win_n)

    try:
        last_loop_t = time.monotonic()
        while rclpy.ok():
            loop_now = time.monotonic()
            loop_step = loop_now - last_loop_t  # real (wall-clock) duration of the last cycle
            last_loop_t = loop_now
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                print(f"\n[Timeout] Maximum mission duration reached ({max_duration:.1f} s).")
                break

            if not robot.is_odom_healthy():
                robot.stop()
                print("\n[Safety Watchdog] Odometry stream frozen or invalid (NaN/stale)! Emergency stop triggered.")
                break

            pos = robot.get_position()
            if len(pos) < 3 or not all(math.isfinite(float(p)) for p in pos[:3]):
                robot.stop()
                print("\n[Safety Watchdog] Non-finite pose received! Emergency stop triggered.")
                break

            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            thr = recast_angle(pos[2])
            e_theta = recast_angle(pos[2] - target[2])

            dist_xy, dist_pose = compute_se2_distance(dx, dy, e_theta)

            x_target_robot = dx * math.cos(thr) + dy * math.sin(thr)
            y_target_robot = -dx * math.sin(thr) + dy * math.cos(thr)

            in_0 = max(-1.0, min(1.0, x_target_robot / X_MAX))
            in_1 = max(-1.0, min(1.0, y_target_robot / Y_MAX))
            in_2 = max(-1.0, min(1.0, e_theta / math.pi))
            cost = in_0 ** 2 + in_1 ** 2 + in_2 ** 2

            cmd_raw = model([in_0, in_1, in_2])
            if is_mecanum:
                out_vx = float(cmd_raw[0])
                out_vy = float(cmd_raw[1])
                out_w = float(cmd_raw[2])
                if not (math.isfinite(out_vx) and math.isfinite(out_vy) and math.isfinite(out_w)):
                    robot.stop()
                    print("\n[Safety Watchdog] Non-finite network output! Emergency stop triggered.")
                    break
                v_cmd, vy_cmd, w_cmd = clamp_mecanum_velocity(
                    vx=out_vx * v_max_effective,
                    vy=out_vy * v_max_effective,
                    w=out_w * w_max_effective,
                    v_max_eff=v_max_effective,
                    w_max_eff=w_max_effective,
                )
            else:
                out_v = float(cmd_raw[0])
                out_w = float(cmd_raw[1])
                if not (math.isfinite(out_v) and math.isfinite(out_w)):
                    robot.stop()
                    print("\n[Safety Watchdog] Non-finite network output! Emergency stop triggered.")
                    break
                v_cmd, vy_cmd, w_cmd = clamp_mecanum_velocity(
                    vx=out_v * v_max_effective,
                    vy=0.0,
                    w=out_w * w_max_effective,
                    v_max_eff=v_max_effective,
                    w_max_eff=w_max_effective,
                )

            pos_window.append((elapsed, pos[0], pos[1], pos[2]))

            in_dock_envelope, is_stationary = check_docking_and_stationarity(
                dist_xy=dist_xy,
                e_theta=e_theta,
                v_cmd=v_cmd,
                w_cmd=w_cmd,
                pos_window=pos_window,
                dist_tol=dist_tol,
                angle_tol=angle_tol,
                vy_cmd=vy_cmd,
                cfg=cfg,
            )

            if in_dock_envelope:
                if not dock_arrival_announced:
                    print(f"\n[Dock Arrival] Robot arrived in dock envelope (d={dist_xy*100:.1f} cm, θ={math.degrees(abs(e_theta)):.1f}°)! Holding for {dwell_time_required:.0f}s dwell test...")
                    dock_arrival_announced = True
            else:
                dock_arrival_announced = False

            if elapsed > 1.0 and in_dock_envelope and is_stationary:
                dwell_timer += loop_step
                if int(dwell_timer) != int(dwell_timer - loop_step) or dwell_timer >= dwell_time_required:
                    print(f"[Dock Dwell] Stationary in envelope: {dwell_timer:5.1f}s / {dwell_time_required:.1f}s (d={dist_xy*100:.1f}cm, θ={math.degrees(abs(e_theta)):.1f}°)")

                if dwell_timer >= dwell_time_required:
                    print(f"\n[Success] Mission Complete! Vehicle continuously docked and stationary for {dwell_timer:.1f} s.")
                    print(f"          Final Position Error: {dist_xy*100:.2f} cm | Heading Error: {math.degrees(abs(e_theta)):.2f}° | SE(2) Pose Metric: {dist_pose*100:.2f} cm")
                    success = True
                    break
            else:
                if dwell_timer > 0.0:
                    print(f"[Notice] Dwell reset! Motion or boundary departure detected at t={elapsed:.1f}s (d={dist_xy*100:.1f}cm).")
                dwell_timer = 0.0

            if int(elapsed / dt) % 20 == 0:
                if is_mecanum:
                    print(f"t={elapsed:5.1f}s | Dist: {dist_xy:5.3f}m | SE(2): {dist_pose:5.3f}m | Head: {math.degrees(abs(e_theta)):5.1f}° | vx: {v_cmd:+5.2f} | vy: {vy_cmd:+5.2f} | w: {w_cmd:+5.2f}")
                else:
                    print(f"t={elapsed:5.1f}s | Dist: {dist_xy:5.3f}m | SE(2): {dist_pose:5.3f}m | Head: {math.degrees(abs(e_theta)):5.1f}° | v_cmd: {v_cmd:+5.2f}m/s | w_cmd: {w_cmd:+5.2f}rad/s")

            robot.set_cmd_vel(v_cmd, w_cmd, v_y=vy_cmd)
            logger.log(position=pos, target=target, command=[v_cmd, w_cmd, vy_cmd])

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[Interrupted] Mission aborted by operator.")
    finally:
        robot.stop()
        try:
            # Artifacts are saved inside a nested try so that the ROS node is always
            # cleaned up, even if saving the CSV / plot / params.json fails.
            now = datetime.now()
            timestamp_folder = now.strftime("test_%Y_%m_%d_%H_%M_%S")

            if output_dir and os.path.basename(os.path.normpath(output_dir)).startswith("test_"):
                test_dir = output_dir if os.path.isabs(output_dir) else os.path.join(_project_root, output_dir)
            else:
                base_eval_dir = output_dir if output_dir else "data/evaluation"
                if not os.path.isabs(base_eval_dir):
                    base_eval_dir = os.path.join(_project_root, base_eval_dir)  # relative to the project root, not the CWD
                test_dir = os.path.join(base_eval_dir, timestamp_folder)

            data_dir = os.path.join(test_dir, "data")
            images_dir = os.path.join(test_dir, "images")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)

            prefix = "eval_success" if success else "eval_stop"
            csv_path = os.path.join(data_dir, f"{prefix}.csv")
            png_path = os.path.join(images_dir, f"{prefix}.png")

            logger.save(csv_path)
            logger.save_plot(png_path)

            final_pos = robot.get_position() if hasattr(robot, "get_position") else [0.0, 0.0, 0.0]
            elapsed_sec = time.time() - start_time
            err_dist = math.hypot(final_pos[0] - target[0], final_pos[1] - target[1])
            err_theta = abs(recast_angle(final_pos[2] - target[2]))
            err_pose = math.hypot(err_dist, HALF_TRACK_B * err_theta)

            params = {
                "test_id": timestamp_folder,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "weights_path": os.path.abspath(weights_path),
                "platform": "physical robot",
                "mode": mode,
                "output_size": output_size,
                "target": target,
                "start_pose": teleport_to,
                "final_position": [round(float(p), 4) for p in final_pos],
                "success": success,
                "elapsed_seconds": round(elapsed_sec, 3),
                "dwell_time_verified_seconds": round(dwell_timer, 2),
                "dwell_time_required_seconds": dwell_time_required,
                "max_duration_seconds": max_duration,
                "error_distance_m": round(err_dist, 4),
                "error_theta_deg": round(math.degrees(err_theta), 2),
                "error_se2_pose_m": round(err_pose, 4),
                "tolerances": {
                    "dist_tol_m": dist_tol,
                    "angle_tol_rad": angle_tol,
                    "half_track_b_m": HALF_TRACK_B,
                },
                "kinematics": {
                    "speed_scale": eff_scale,
                    "v_max_nominal_mps": V_MAX,
                    "v_max_effective_mps": v_max_effective,
                    "w_max_radps": w_max_effective,
                    "x_max_m": X_MAX,
                    "y_max_m": Y_MAX,
                    "dt_sec": dt,
                },
                "artifacts": {
                    "folder": os.path.abspath(test_dir),
                    "params_json": os.path.abspath(os.path.join(test_dir, "params.json")),
                    "data_dir": os.path.abspath(data_dir),
                    "images_dir": os.path.abspath(images_dir),
                    "csv_path": os.path.abspath(csv_path),
                    "png_path": os.path.abspath(png_path)
                }
            }
            params_path = os.path.join(test_dir, "params.json")
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            print(f"[{'Success' if success else 'Saved'}] Trial metadata and telemetry preserved in: {params_path}")
        finally:
            robot.cleanup()
            rclpy.shutdown()

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-Loop Autonomous Positioning & Metrological Docking.")
    parser.add_argument("--mode", choices=["diff", "mecanum"], default="mecanum", help="Kinematic mode: 'mecanum' (default) or 'diff' (requires --weights of a differential model)")
    parser.add_argument("--weights", type=str, default=None, help="Path to synaptic weights JSON")
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Target pose x y theta")
    parser.add_argument("--teleport-to", "--start", nargs=3, type=float, default=None, dest="teleport_to", help="Starting pose x y theta")
    parser.add_argument("--real", action="store_true", help="Physical robot (required: no Gazebo model exists for the Mecanum LIMO)")
    parser.add_argument("--max-time", type=float, default=None, help="Maximum duration ceiling (s)")
    parser.add_argument("--dist-tol", type=float, default=None, help="Docking positioning tolerance (m)")
    parser.add_argument("--angle-tol", type=float, default=None, help="Docking orientation tolerance (rad)")
    parser.add_argument("--dwell-time", type=float, default=None, help="Required stationary dwell time at dock (s)")
    parser.add_argument("--outdir", type=str, default="data/evaluation", help="Destination telemetry directory")
    parser.add_argument("--odom-topic", type=str, default=None, help="Odometry topic name")
    parser.add_argument("--cmd-vel-topic", type=str, default=None, help="Command velocity topic name")
    parser.add_argument("--speed-scale", type=float, default=None, help="Override kinematics.speed_scale from config.json (clamped to [0.2, 1.0])")
    args = parser.parse_args()

    run_autonomous_session(
        weights_path=args.weights,
        target=args.target,
        teleport_to=args.teleport_to,
        is_real=args.real,
        max_duration=args.max_time,
        dist_tol=args.dist_tol,
        angle_tol=args.angle_tol,
        dwell_time_required=args.dwell_time,
        output_dir=args.outdir,
        odom_topic=args.odom_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        speed_scale=args.speed_scale,
        mode=args.mode,
    )
