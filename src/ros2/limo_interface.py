"""
ROS 2 Hardware and Simulation Interface for AgileX LIMO (Mecanum & Diff)
========================================================================
Handles asynchronous communication with the physical robot or Gazebo simulator
via standard /odom and /cmd_vel topics.

Features:
- Subscription to odometry topic (/odom, /wheel/odom) at 20-50 Hz.
- Strict IEEE-754 NaN/Inf filtering on incoming odometry and outgoing commands.
- Real-time odometry staleness watchdog (500 ms default from config.json).
- Vectorial linear velocity limiting sqrt(vx^2 + vy^2) <= V_MAX.
- Verified Gazebo teleportation (never forges pose offset if teleport fails).

Project: LIMO Offline Learning (CERV / ENIB)
"""

from __future__ import annotations

import os
import sys
import math
import time
import threading
from typing import List, Optional

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.config import load_config

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    ROS2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ROS2_AVAILABLE = False
    Node = object  # Fallback for non-ROS environments
    Twist = object
    Odometry = object


def sanitize_and_clamp_twist(
    v_lin: float,
    v_ang: float,
    v_y: float = 0.0,
    v_max: float = 0.50,
    w_max: float = 1.00,
    enforce_limits: bool = True,
) -> tuple[float, float, float]:
    """
    Sanitizes velocity commands against NaN/Inf (converting invalid values to 0.0)
    and clamps translational speed by Euclidean vector norm sqrt(vx^2 + vy^2) <= v_max.
    """
    try:
        vx = float(v_lin)
        vy = float(v_y)
        w = float(v_ang)
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0

    if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(w)):
        return 0.0, 0.0, 0.0

    if enforce_limits:
        v_mag = math.hypot(vx, vy)
        if v_mag > v_max and v_mag > 0.0:
            scale = v_max / v_mag
            vx *= scale
            vy *= scale
        w = max(-w_max, min(w_max, w))

    return vx, vy, w


class LimoROS2Interface(Node):
    """
    Isolated ROS 2 interface for the AgileX LIMO mobile robot.

    Provides a strict method contract:
    - get_position() -> [x, y, theta]
    - is_odom_healthy(max_age_s) -> bool
    - set_cmd_vel(v_lin, v_ang, v_y=0.0)
    - stop()
    - cleanup()
    """

    def __init__(
        self,
        safety_limits: bool = True,
        odom_topic: Optional[str] = None,
        cmd_vel_topic: Optional[str] = None,
        odom_watchdog_timeout_s: Optional[float] = None,
    ):
        if not ROS2_AVAILABLE:
            raise RuntimeError("The rclpy module is not available in this Python environment.")

        super().__init__("limo_offline_interface")
        cfg = load_config()
        kin_cfg = cfg["kinematics"]
        ros_cfg = cfg["ros2_and_safety"]

        self.v_max = float(kin_cfg["v_max"])
        self.w_max = float(kin_cfg["w_max"])
        self.safety_limits = safety_limits
        self.odom_topic = odom_topic if odom_topic is not None else ros_cfg["odom_topic"]
        self.cmd_vel_topic = cmd_vel_topic if cmd_vel_topic is not None else ros_cfg["cmd_vel_topic"]
        self.odom_watchdog_timeout_s = float(
            odom_watchdog_timeout_s
            if odom_watchdog_timeout_s is not None
            else ros_cfg["odom_watchdog_timeout_s"]
        )
        self.default_gazebo_model = str(ros_cfg["gazebo_model_name"])
        self._active_odom_topic: Optional[str] = None

        # Internal robot pose [x, y, theta] and origin offsets
        self._position = [0.0, 0.0, 0.0]
        self._raw_position = [0.0, 0.0, 0.0]
        self._raw_origin = [0.0, 0.0, 0.0]
        self._target_origin = [0.0, 0.0, 0.0]
        self._offset_th = 0.0
        self._offset = [0.0, 0.0, 0.0]
        self._lock = threading.Lock()
        self._odom_received = False
        self._odom_valid = False
        self._last_odom_time = 0.0
        self._odom_seq = 0

        # Odometry subscriptions
        self._odom_subs = []
        if self.odom_topic and self.odom_topic != "auto":
            sub = self.create_subscription(
                Odometry, self.odom_topic, lambda msg, t=self.odom_topic: self._odom_callback(msg, t), 10
            )
            self._odom_subs.append(sub)
            print(f"[LimoROS2Interface] Subscribed to odometry topic: {self.odom_topic}")
        else:
            for candidate in ["/odom", "/wheel/odom"]:
                sub = self.create_subscription(
                    Odometry, candidate, lambda msg, t=candidate: self._odom_callback(msg, t), 10
                )
                self._odom_subs.append(sub)
            print("[LimoROS2Interface] Auto-detecting odometry stream (/odom and /wheel/odom)...")

        self._cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Asynchronous background thread for ROS 2 spinning
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._spin_thread.start()

        # Await first valid odometry frame
        wait_start = time.time()
        while not self._odom_received and rclpy.ok():
            time.sleep(0.05)
            if time.time() - wait_start > 5.0:
                topics_str = self.odom_topic if (self.odom_topic and self.odom_topic != "auto") else "/odom, /wheel/odom"
                print(f"[LimoROS2Interface] Waiting for valid odometry stream on {topics_str}...")
                wait_start = time.time()

        if not self._odom_received:
            raise RuntimeError("[LimoROS2Interface] ROS 2 was shut down before any valid odometry frame was received.")
        print(f"[LimoROS2Interface] First valid odometry frame received on '{self.odom_topic}'.")

    def _odom_callback(self, msg: Odometry, topic_name: str = "/odom") -> None:
        """
        Odometry message processing callback with strict NaN/Inf and quaternion validation.
        """
        try:
            pos_x = float(msg.pose.pose.position.x)
            pos_y = float(msg.pose.pose.position.y)
            quat = msg.pose.pose.orientation
            qx = float(quat.x)
            qy = float(quat.y)
            qz = float(quat.z)
            qw = float(quat.w)
        except (AttributeError, TypeError, ValueError):
            with self._lock:
                self._odom_valid = False
            return

        if not all(math.isfinite(v) for v in (pos_x, pos_y, qx, qy, qz, qw)):
            with self._lock:
                self._odom_valid = False
            return

        q_norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
        if q_norm_sq < 1e-8:
            with self._lock:
                self._odom_valid = False
            return

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        theta = math.atan2(siny_cosp, cosy_cosp)
        if not math.isfinite(theta):
            with self._lock:
                self._odom_valid = False
            return

        with self._lock:
            if self._active_odom_topic is None:
                self._active_odom_topic = topic_name
                self.odom_topic = topic_name
                print(f"[LimoROS2Interface] Odometry stream locked to topic: {topic_name}")
            elif self._active_odom_topic != topic_name:
                return

            self._raw_position = [pos_x, pos_y, theta]
            dx_raw = pos_x - self._raw_origin[0]
            dy_raw = pos_y - self._raw_origin[1]
            diff_th = self._offset_th
            dx_world = dx_raw * math.cos(diff_th) - dy_raw * math.sin(diff_th)
            dy_world = dx_raw * math.sin(diff_th) + dy_raw * math.cos(diff_th)
            adj_theta = math.atan2(
                math.sin(theta + diff_th),
                math.cos(theta + diff_th)
            )
            self._position = [
                self._target_origin[0] + dx_world,
                self._target_origin[1] + dy_world,
                adj_theta
            ]
            self._odom_received = True
            self._odom_valid = True
            self._last_odom_time = time.monotonic()
            self._odom_seq += 1

    def is_odom_healthy(self, max_age_s: Optional[float] = None) -> bool:
        """
        Verifies that odometry is valid (finite) and fresh (not frozen/stopped).
        """
        timeout = float(max_age_s if max_age_s is not None else self.odom_watchdog_timeout_s)
        with self._lock:
            if not self._odom_received or not self._odom_valid:
                return False
            age = time.monotonic() - self._last_odom_time
            if age > timeout:
                return False
            return all(math.isfinite(p) for p in self._position)

    def reset_pose(self, x0: float, y0: float, theta0: float) -> None:
        """
        Calibrates the odometry origin so the robot starts at (x0, y0, theta0).
        """
        if not all(math.isfinite(v) for v in (x0, y0, theta0)):
            raise ValueError(f"Invalid non-finite reset_pose target: ({x0}, {y0}, {theta0})")
        with self._lock:
            raw_x, raw_y, raw_th = self._raw_position
            diff_th = math.atan2(math.sin(theta0 - raw_th), math.cos(theta0 - raw_th))
            self._raw_origin = [raw_x, raw_y, raw_th]
            self._target_origin = [x0, y0, theta0]
            self._offset_th = diff_th
            self._offset = [x0 - raw_x, y0 - raw_y, diff_th]
            self._position = [x0, y0, theta0]
        print(f"[LimoROS2Interface] Pose calibrated to: x={x0:.2f} m, y={y0:.2f} m, theta={theta0:.2f} rad")

    def teleport_gazebo(self, x0: float, y0: float, theta0: float, model_name: Optional[str] = None) -> bool:
        """
        Teleports the 3D robot model in Gazebo and verifies success before calibrating odometry.
        Never forges the software pose if the simulator teleportation fails.
        """
        import subprocess
        if not all(math.isfinite(v) for v in (x0, y0, theta0)):
            print(f"[LimoROS2Interface] ERROR: Refusing to teleport to non-finite pose ({x0}, {y0}, {theta0}).")
            return False

        target_model = model_name or self.default_gazebo_model
        candidate_models = []
        for m in [target_model, "limo_description", "limo", "agilex_limo"]:
            if m and m not in candidate_models:
                candidate_models.append(m)

        self.stop()
        time.sleep(0.10)

        success = False
        # 1. Attempt via gz model CLI utility
        for cand in candidate_models:
            try:
                res = subprocess.run(
                    ["gz", "model", "-m", cand, "-x", str(x0), "-y", str(y0), "-z", "0.05", "-Y", str(theta0)],
                    capture_output=True, text=True, timeout=2.0
                )
                stderr_low = (res.stderr or "").lower()
                if res.returncode == 0 and "error" not in stderr_low and "unable" not in stderr_low:
                    success = True
                    break
            except Exception:
                pass

        # 2. Attempt via ROS 2 /set_entity_state service
        if not success:
            qz = math.sin(theta0 / 2.0)
            qw = math.cos(theta0 / 2.0)
            for cand in candidate_models:
                try:
                    res = subprocess.run(
                        [
                            "ros2", "service", "call", "/set_entity_state", "gazebo_msgs/srv/SetEntityState",
                            f"{{state: {{name: '{cand}', pose: {{position: {{x: {x0}, y: {y0}, z: 0.05}}, orientation: {{z: {qz}, w: {qw}}}}}}}}}"
                        ],
                        capture_output=True, text=True, timeout=3.0
                    )
                    out_low = ((res.stdout or "") + " " + (res.stderr or "")).lower()
                    if res.returncode == 0 and ("success=true" in out_low or "success: true" in out_low):
                        success = True
                        break
                except Exception:
                    pass

        if not success:
            self.stop()
            print(
                f"[LimoROS2Interface] ERROR: Gazebo teleportation to "
                f"(x={x0:.2f}, y={y0:.2f}, theta={theta0:.2f}) FAILED. Pose was NOT forged."
            )
            return False

        # 3. Physical settling and verified odometric calibration
        time.sleep(0.35)
        self.stop()
        if not self.is_odom_healthy():
            print("[LimoROS2Interface] ERROR: Odometry stream unhealthy after teleportation.")
            return False

        self.reset_pose(x0, y0, theta0)
        return True

    def get_position(self) -> List[float]:
        """
        Returns the current Cartesian and angular pose of the robot [x, y, theta].
        """
        with self._lock:
            return list(self._position)

    def set_cmd_vel(self, v_lin: float, v_ang: float, v_y: float = 0.0) -> None:
        """
        Publishes linear (x, optional y for mecanum) and angular velocity commands on /cmd_vel.
        Strictly filters NaN/Inf to 0.0 and clamps translational velocity by Euclidean norm.
        """
        vx, vy, w = sanitize_and_clamp_twist(
            v_lin=v_lin,
            v_ang=v_ang,
            v_y=v_y,
            v_max=self.v_max,
            w_max=self.w_max,
            enforce_limits=self.safety_limits,
        )

        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = w
        self._cmd_pub.publish(msg)

    def stop(self) -> None:
        """Immediately sends an emergency stop command (zero velocities)."""
        self.set_cmd_vel(0.0, 0.0, 0.0)

    def cleanup(self) -> None:
        """Stops the robot and cleanly destroys the ROS 2 node."""
        self.stop()
        time.sleep(0.05)
        self.destroy_node()
