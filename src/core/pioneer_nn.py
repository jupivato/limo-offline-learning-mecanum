"""
PioneerNN Neural Controller Architecture
========================================
Multi-Layer Perceptron used as the docking controller of the AgileX LIMO:

    y = tanh(W_o · tanh(W_h · x + b_h) + b_o)

Inputs (robot frame, normalized to [-1, 1]):
    in_0 = x_target_robot / x_max
    in_1 = y_target_robot / y_max
    in_2 = recast(theta_robot - theta_target) / pi
Outputs in [-1, 1], scaled at execution time by v_max * speed_scale and w_max
(2 outputs for the differential robot, 3 for the Mecanum robot).

Exact stationarity at the dock: MLP(0, 0, 0) = 0, guaranteed by bias-free layers
(tanh(0) = 0) or, with bias, by the output-bias synchronization
b_o = -W_o · tanh(b_h) performed during training.

Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB)
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn


def recast_angle(angle_rad: float) -> float:
    """Recasts an arbitrary angle into the principal interval (-pi, +pi]."""
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


class PioneerNN(nn.Module):
    """PioneerNN MLP (one tanh hidden layer, tanh outputs) for mobile robot velocity control."""

    def __init__(self, input_size: int = 3, hidden_size: int = 1000, output_size: int = 2, use_bias: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.hidden = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.output = nn.Linear(hidden_size, output_size, bias=use_bias)
        self.activation = nn.Tanh()
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Default multiscale initialization (random directions, log-spaced scales).
        The training scripts re-initialize the weights with their own seeded scheme.
        """
        with torch.no_grad():
            dirs = torch.randn(self.hidden_size, self.input_size)
            norms = torch.norm(dirs, dim=1, keepdim=True).clamp(min=1e-6)
            scales = torch.logspace(-0.2, 1.85, self.hidden_size).unsqueeze(1)
            self.hidden.weight.copy_((dirs / norms) * scales)
            nn.init.uniform_(self.output.weight, -1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size))
            if self.use_bias:
                nn.init.zeros_(self.hidden.bias)
                nn.init.zeros_(self.output.bias)

    def forward(self, x: Any) -> torch.Tensor:
        """y = tanh(W_o · tanh(W_h · x + b_h) + b_o). Accepts a tensor or a (nested) list."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.device != self.hidden.weight.device:
            x = x.to(self.hidden.weight.device)
        if not self.training:
            with torch.no_grad():
                return self.activation(self.output(self.activation(self.hidden(x))))
        return self.activation(self.output(self.activation(self.hidden(x))))

    def save_weights_to_json(self) -> Dict[str, Any]:
        """Serializes the weights and architecture metadata into the project JSON format."""
        data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "use_bias": self.use_bias,
            "input_weights": self.hidden.weight.detach().cpu().t().tolist(),
            "output_weights": self.output.weight.detach().cpu().t().tolist(),
        }
        if self.use_bias:
            data["hidden_bias"] = self.hidden.bias.detach().cpu().tolist()
            data["output_bias"] = self.output.bias.detach().cpu().tolist()
        return data

    def load_weights_from_json(self, json_obj: Dict[str, Any]) -> None:
        """Loads weights from a JSON object; any size mismatch raises an error."""
        meta_input = json_obj.get("input_size", len(json_obj["input_weights"]))
        meta_hidden = json_obj.get("hidden_size", self.hidden_size)
        meta_output = json_obj.get("output_size", self.output_size)

        if meta_input != self.input_size or len(json_obj["input_weights"]) != self.input_size:
            raise ValueError(f"Input layer mismatch: JSON={meta_input}, Model={self.input_size}")
        if meta_hidden != self.hidden_size:
            raise ValueError(f"Hidden layer mismatch: JSON={meta_hidden}, Model={self.hidden_size}")
        if meta_output != self.output_size:
            raise ValueError(f"Output layer mismatch: JSON={meta_output}, Model={self.output_size}")

        self.use_bias = bool(json_obj.get("use_bias", False))
        self.hidden = nn.Linear(self.input_size, self.hidden_size, bias=self.use_bias)
        self.output = nn.Linear(self.hidden_size, self.output_size, bias=self.use_bias)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor(json_obj["input_weights"], dtype=torch.float32).t())
            self.output.weight.copy_(torch.tensor(json_obj["output_weights"], dtype=torch.float32).t())
            if self.use_bias:
                self.hidden.bias.copy_(torch.tensor(json_obj.get("hidden_bias", [0.0] * self.hidden_size), dtype=torch.float32))
                self.output.bias.copy_(torch.tensor(json_obj.get("output_bias", [0.0] * self.output_size), dtype=torch.float32))

    def save_to_json_file(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.save_weights_to_json(), f, indent=2)

    def load_from_json_file(self, file_path: str) -> None:
        """
        Loads synaptic weights from exactly the requested file (absolute path, or
        path relative to the current directory or to the project root). There is
        no silent fallback to another weights file: a missing file raises.
        """
        candidates = [file_path]
        if not os.path.isabs(file_path):
            candidates.append(os.path.join(_project_root, file_path))
        resolved_path = next((c for c in candidates if os.path.exists(c)), None)
        if resolved_path is None:
            raise FileNotFoundError(
                f"Synaptic weights file not found: {file_path} "
                f"(also tried {os.path.join(_project_root, file_path)})"
            )
        resolved_path = os.path.abspath(resolved_path)
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_weights_from_json(data)
        print(f"[PioneerNN] Weights loaded from: {resolved_path}")
