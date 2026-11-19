"""
PioneerNN Neural Controller Architecture
=========================================
Nonlinear Multi-Layer Perceptron (MLP) for closed-loop pose positioning 
and docking of non-holonomic and holonomic mobile robots (AgileX LIMO).

Theoretical Framework:
----------------------
Based on the canonical ENIB / CERV / Lab-STICC formulation (Prof. Patrick Hénaff):
1. Robot-Centric Kinematic Mapping:
   The state vector is defined in the robot's local frame of reference:
       in_0 = x_target_robot / X_MAX ∈ [-1.0, +1.0]
       in_1 = y_target_robot / Y_MAX ∈ [-1.0, +1.0]
       in_2 = recast(theta_robot - theta_target) / π ∈ [-1.0, +1.0]
   providing total translation and rotation invariance with respect to global world frames.

2. Equilibrium & Stationarity Architecture:
   - Multiscale hidden representation across all 1,000 hidden units.
   - Exact stationarity at the origin x = [0, 0, 0]^T ==> y = 0: guaranteed by
     bias-free layers (tanh(0) = 0) or, with bias, by the output-bias sync
     b_o = -W_o * tanh(b_h) performed during training.

3. Radial Angular Damping Gain (K_theta) -- REFERENCE ONLY:
       K_theta(x_R, y_R) = 1.0 - 0.9 * exp(-10 * (x_R^2 + y_R^2)^2) ∈ [0.1, 1.0]
   `compute_k_theta` is kept for documentation/comparison purposes but is NOT
   applied anywhere in the control loop: execution is 100% pure neural policy
   (v = out_v * v_max * speed_scale, w = out_w * w_max).

Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB)
Standards: ROS REP-103 & ISO 18646-2
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys
import math
from typing import Dict, Any, Union, List, Optional, Tuple

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
TORCH_AVAILABLE = True


def recast_angle(angle_rad: float) -> float:
    """
    Recasts an arbitrary angle into the canonical principal interval (-pi, +pi].
    """
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def compute_theta_strategic(x: float, y: float) -> float:
    """
    Computes the canonical ENIB strategic approach orientation theta_s(x, y).
    """
    return math.tanh(10.0 * float(x)) * math.atan(1.0 * float(y))


def compute_k_theta(x_r: float, y_r: float) -> float:
    """
    Computes the radial angular velocity damping gain K_theta in [0.1, 1.0].
    Reference only: not used by the execution / benchmark control loops.
    """
    try:
        r2 = float(x_r) ** 2 + float(y_r) ** 2
        r4 = r2 ** 2
        arg = -10.0 * r4
        if arg < -50.0:
            arg = -50.0
        elif arg > 0.0:
            arg = 0.0
        k_theta = 1.0 - 0.9 * math.exp(arg)
    except Exception:
        k_theta = 1.0
    return max(0.1, min(1.0, float(k_theta)))


class ReplayBuffer:
    """
    Transition replay buffer for mobile robot reinforcement and imitation learning.
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: collections.deque[Tuple[List[float], List[float], float, List[float], bool]] = collections.deque(maxlen=capacity)

    def add(
        self,
        state: Union[List[float], Any],
        action: Union[List[float], Any],
        reward: float,
        next_state: Union[List[float], Any],
        done: bool
    ) -> None:
        self.buffer.append((list(state), list(action), float(reward), list(next_state), bool(done)))

    def sample(self, batch_size: int) -> Dict[str, Any]:
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "states": list(states),
            "actions": list(actions),
            "rewards": list(rewards),
            "next_states": list(next_states),
            "dones": list(dones)
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


class DummyTensor(list):
    """Tensor-compatible list container for pure-Python fallback execution."""
    def squeeze(self):
        return self

    def tolist(self):
        return list(self)


if TORCH_AVAILABLE:
    class BaseNN(nn.Module):
        pass
else:
    class BaseNN(object):
        pass


class PioneerNN(BaseNN):
    """
    PioneerNN Multi-Layer Perceptron (MLP) for Mobile Robot Velocity Control.
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 1000,
        output_size: int = 2,
        use_bias: bool = False
    ):
        if TORCH_AVAILABLE:
            super(PioneerNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_bias = use_bias

        if TORCH_AVAILABLE:
            self.hidden = nn.Linear(input_size, hidden_size, bias=use_bias)
            self.output = nn.Linear(hidden_size, output_size, bias=use_bias)
            self.activation = nn.Tanh()
            self._init_weights()

        self._py_input_weights = [[0.0] * hidden_size for _ in range(input_size)]
        self._py_output_weights = [[0.0] * output_size for _ in range(hidden_size)]
        self._py_hidden_bias = [0.0] * hidden_size
        self._py_output_bias = [0.0] * output_size

    def _init_weights(self) -> None:
        """
        Initializes synaptic weights with multiscale receptive fields across all hidden units
        so that all 1,000 neurons participate across far-field, braking, and terminal zones.
        """
        if TORCH_AVAILABLE:
            with torch.no_grad():
                dirs = torch.randn(self.hidden_size, self.input_size)
                norms = torch.norm(dirs, dim=1, keepdim=True).clamp(min=1e-6)
                scales = torch.logspace(-0.2, 1.85, self.hidden_size).unsqueeze(1)
                self.hidden.weight.copy_((dirs / norms) * scales)
                nn.init.uniform_(self.output.weight, -1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size))
                if self.use_bias:
                    nn.init.zeros_(self.hidden.bias)
                    nn.init.zeros_(self.output.bias)

    def eval(self):
        if TORCH_AVAILABLE and hasattr(super(), "eval"):
            super().eval()
        return self

    def train(self, mode: bool = True):
        if TORCH_AVAILABLE and hasattr(super(), "train"):
            super().train(mode)
        return self

    def forward(self, x: Any) -> Any:
        """
        Executes forward propagation pass: y = Tanh(W_out * Tanh(W_in * x + b_h) + b_o).
        """
        if TORCH_AVAILABLE:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.device != self.hidden.weight.device:
                x = x.to(self.hidden.weight.device)
            if not self.training:
                with torch.no_grad():
                    out = self.activation(self.hidden(x))
                    out = self.activation(self.output(out))
                    return out
            out = self.activation(self.hidden(x))
            out = self.activation(self.output(out))
            return out

        if hasattr(x, "tolist"):
            in_vec = x.tolist()
        else:
            in_vec = list(x)

        if len(in_vec) > 0 and isinstance(in_vec[0], list):
            in_vec = in_vec[0]

        h = []
        for j in range(self.hidden_size):
            val = self._py_hidden_bias[j] if self.use_bias else 0.0
            val += sum(in_vec[i] * self._py_input_weights[i][j] for i in range(self.input_size))
            h.append(math.tanh(val))

        out = []
        for k in range(self.output_size):
            val = self._py_output_bias[k] if self.use_bias else 0.0
            val += sum(h[j] * self._py_output_weights[j][k] for j in range(self.hidden_size))
            out.append(math.tanh(val))

        return DummyTensor(out)

    def __call__(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(super(), "__call__"):
            return super().__call__(x)
        return self.forward(x)

    def save_weights_to_json(self) -> Dict[str, Any]:
        """
        Serializes synaptic weights and architecture metadata into the canonical JSON dictionary.
        """
        if TORCH_AVAILABLE:
            input_weights = self.hidden.weight.detach().cpu().t().tolist()
            output_weights = self.output.weight.detach().cpu().t().tolist()

            data = {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "use_bias": self.use_bias,
                "input_weights": input_weights,
                "output_weights": output_weights,
            }

            if self.use_bias:
                data["hidden_bias"] = self.hidden.bias.detach().cpu().tolist()
                data["output_bias"] = self.output.bias.detach().cpu().tolist()

            return data

        data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "use_bias": self.use_bias,
            "input_weights": self._py_input_weights,
            "output_weights": self._py_output_weights,
        }
        if self.use_bias:
            data["hidden_bias"] = self._py_hidden_bias
            data["output_bias"] = self._py_output_bias
        return data

    def load_weights_from_json(self, json_obj: Dict[str, Any]) -> None:
        """
        Loads synaptic weights from a JSON object.
        """
        meta_hidden = json_obj.get("hidden_size", self.hidden_size)
        meta_output = json_obj.get("output_size", self.output_size)
        file_bias = json_obj.get("use_bias", False)

        if meta_hidden != self.hidden_size:
            raise ValueError(f"Hidden layer mismatch: JSON={meta_hidden}, Model={self.hidden_size}")
        if meta_output != self.output_size:
            raise ValueError(f"Output layer mismatch: JSON={meta_output}, Model={self.output_size}")

        self.use_bias = file_bias

        if TORCH_AVAILABLE:
            self.hidden = nn.Linear(self.input_size, self.hidden_size, bias=self.use_bias)
            self.output = nn.Linear(self.hidden_size, self.output_size, bias=self.use_bias)

            src_in_weights = json_obj["input_weights"]
            copy_rows = min(len(src_in_weights), self.input_size)
            input_weights = torch.zeros(self.input_size, self.hidden_size)
            input_weights[:copy_rows, :] = torch.tensor(src_in_weights[:copy_rows], dtype=torch.float32)

            output_weights = torch.tensor(json_obj["output_weights"], dtype=torch.float32)

            with torch.no_grad():
                self.hidden.weight.copy_(input_weights.t())
                self.output.weight.copy_(output_weights.t())

                if self.use_bias:
                    if "hidden_bias" in json_obj:
                        self.hidden.bias.copy_(torch.tensor(json_obj["hidden_bias"], dtype=torch.float32))
                    else:
                        nn.init.zeros_(self.hidden.bias)

                    if "output_bias" in json_obj:
                        self.output.bias.copy_(torch.tensor(json_obj["output_bias"], dtype=torch.float32))
                    else:
                        nn.init.zeros_(self.output.bias)

        src_in = json_obj["input_weights"]
        py_in = []
        for i in range(self.input_size):
            if i < len(src_in):
                py_in.append(list(src_in[i]))
            else:
                py_in.append([0.0] * self.hidden_size)

        self._py_input_weights = py_in
        self._py_output_weights = json_obj["output_weights"]
        self._py_hidden_bias = json_obj.get("hidden_bias", [0.0] * self.hidden_size)
        self._py_output_bias = json_obj.get("output_bias", [0.0] * self.output_size)

    def save_to_json_file(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = self.save_weights_to_json()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json_file(self, file_path: str) -> None:
        resolved_path = file_path
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.join(_project_root, file_path),
                os.path.join(_project_root, "models", "pioneer_mecanum_weights.json"),
                os.path.join(_project_root, "models", "supervised_w_torch_diff.json"),
                os.path.join(_project_root, "models", "pioneer_model_weights.json"),
                os.path.join(os.path.dirname(os.path.abspath(file_path)), "pioneer_mecanum_weights.json"),
                os.path.join(os.path.dirname(os.path.abspath(file_path)), "supervised_w_torch_diff.json"),
                "models/pioneer_mecanum_weights.json",
                "models/supervised_w_torch_diff.json",
                "models/pioneer_model_weights.json",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    resolved_path = cand
                    break

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Synaptic weights file not found: {file_path}")
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_weights_from_json(data)
