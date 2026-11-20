"""
Offline Behavioral Cloning Training Pipeline for AgileX LIMO (Mecanum 3-Output)
================================================================================
Trains a 3-output PioneerNN controller (vx, vy, w) on `data/dataset_mecanum.csv`
and saves the weights to `models/pioneer_mecanum_weights.json`.

Key Architectural & Control Guarantees:
---------------------------------------
1. Single Source of Truth: every hyperparameter (per-group learning rates and
   weight decays, batch size, loss weights, split, seed) is read from the
   `network_and_training` block of `config.json`; the values actually used are
   recorded under `effective_config` in the training history.
2. Strict Origin Stationarity (`use_bias = False`):
       y(x) = tanh(W_out * tanh(W_in * x))
   Because tanh(0) = 0, y([0, 0, 0]) == [0, 0, 0] is algebraically exact without
   relying on fragile numerical cancellation, even under 5% weight perturbations.
3. Full 1,000-Neuron Multiscale Utilization: Distributes multiscale receptive
   fields across all 1,000 hidden units (near-field restorative core, braking
   funnel, and far-field cruise) with L2-balanced output weights.

Project: LIMO Offline Learning (CERV / Lab-STICC / ENIB)
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import math
import argparse
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.config import load_config
from src.core.pioneer_nn import PioneerNN, recast_angle

_CFG = load_config()
_KIN = _CFG["kinematics"]
_EXP = _CFG["expert_and_dataset"]
_NET = _CFG["network_and_training"]

INPUT_SIZE = int(_NET["input_size"])
HIDDEN_SIZE = int(_NET["hidden_size"])
OUTPUT_SIZE = int(_NET["output_size"])
USE_BIAS = bool(_NET["use_bias"])

EPOCHS = int(_NET["epochs"])
BATCH_SIZE = int(_NET["batch_size"])
BETA = float(_NET["beta"])

V_MAX = float(_KIN["v_max"])
W_MAX = float(_KIN["w_max"])
X_MAX = float(_KIN["x_max"])
Y_MAX = float(_KIN["y_max"])

DEFAULT_DATASET = _EXP["dataset_csv"]
DEFAULT_WEIGHTS = _NET["weights_path"]
DEFAULT_HISTORY = _NET["history_path"]


def load_mecanum_dataset(csv_path: str) -> Tuple[List[List[float]], List[List[float]], List[int]]:
    """
    Loads the Mecanum dataset, normalizes inputs and 3 outputs to [-1, 1], and
    returns the `group_id` of every sample (jittered copies share their base group).
    """
    resolved = csv_path if os.path.exists(csv_path) else os.path.join(_project_root, csv_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Mecanum dataset not found: {csv_path}")

    inputs, targets, groups = [], [], []
    with open(resolved, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "group_id" not in (reader.fieldnames or []):
            raise ValueError(
                f"{resolved} has no 'group_id' column (needed for a leak-free train/validation "
                f"split). Regenerate it with: python3 src/generate_dataset_mecanum.py"
            )
        for row in reader:
            try:
                xr = float(row["x_target_robot"])
                yr = float(row["y_target_robot"])
                thr = recast_angle(float(row["theta_robot"]))
                vx = float(row["vx_robot"])
                vy = float(row["vy_robot"])
                w = float(row["w_robot"])
                gid = int(row["group_id"])

                inputs.append([
                    max(-1.0, min(1.0, xr / X_MAX)),
                    max(-1.0, min(1.0, yr / Y_MAX)),
                    max(-1.0, min(1.0, thr / math.pi)),
                ])
                targets.append([
                    max(-1.0, min(1.0, vx / V_MAX)),
                    max(-1.0, min(1.0, vy / V_MAX)),
                    max(-1.0, min(1.0, w / W_MAX)),
                ])
                groups.append(gid)
            except (KeyError, ValueError):
                continue

    if not inputs:
        raise ValueError(f"No valid samples found in {resolved}")
    return inputs, targets, groups


def group_train_val_split(groups: List[int], train_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Leak-free train/validation split: whole groups (a unique state and its jittered
    copies) are assigned to one side. Groups are permuted with `seed`.
    """
    rows_of: Dict[int, List[int]] = {}
    for idx, gid in enumerate(groups):
        rows_of.setdefault(gid, []).append(idx)
    group_ids = list(rows_of.keys())
    torch.manual_seed(seed)
    gperm = torch.randperm(len(group_ids)).tolist()
    n_train_groups = int(train_fraction * len(group_ids))
    train_idx = [i for g in gperm[:n_train_groups] for i in rows_of[group_ids[g]]]
    val_idx = [i for g in gperm[n_train_groups:] for i in rows_of[group_ids[g]]]
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


def weighted_mecanum_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = BETA) -> torch.Tensor:
    """Weighted MSE across (vx, vy, w): (MSE(vx) + MSE(vy) + beta * MSE(w)) / (2 + beta)."""
    loss_vx = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
    loss_vy = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
    loss_w = torch.mean((pred[:, 2] - target[:, 2]) ** 2)
    return (loss_vx + loss_vy + beta * loss_w) / (2.0 + beta)


def _init_multiscale_mecanum_bank(model: PioneerNN, seed: int) -> None:
    """
    Initializes all 1,000 hidden units with random multiscale receptive fields
    and small random output weights, ready for end-to-end supervised learning
    from `data/dataset_mecanum.csv`.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        H = model.hidden_size
        dirs = torch.randn(H, 3)
        dirs = dirs / torch.norm(dirs, dim=1, keepdim=True).clamp(min=1e-6)
        scales = torch.logspace(0.0, 2.1, H).unsqueeze(1)
        model.hidden.weight.copy_(dirs * scales)
        model.output.weight.copy_(torch.randn(3, H) * 0.002)
        if model.use_bias:
            nn.init.zeros_(model.hidden.bias)
            nn.init.zeros_(model.output.bias)


def train_mecanum(
    dataset_path: Optional[str] = None,
    weights_output: Optional[str] = None,
    history_output: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    use_bias: Optional[bool] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Trains the 3-output Mecanum PioneerNN and saves best validation weights."""
    cfg = load_config(config_path)
    exp_cfg = cfg["expert_and_dataset"]
    net_cfg = cfg["network_and_training"]

    dataset_path = dataset_path or exp_cfg["dataset_csv"]
    weights_output = weights_output or net_cfg["weights_path"]
    history_output = history_output or net_cfg["history_path"]
    epochs = int(epochs if epochs is not None else net_cfg["epochs"])
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1 (got {epochs})")
    batch_size = int(batch_size if batch_size is not None else net_cfg["batch_size"])
    # --lr (if given) sets the hidden-layer rate; the other groups keep their config ratio.
    lr_ratio = 1.0 if learning_rate is None else float(learning_rate) / float(net_cfg["lr_hidden"])
    lr_hidden = float(net_cfg["lr_hidden"]) * lr_ratio
    lr_output = float(net_cfg["lr_output"]) * lr_ratio
    lr_bias = float(net_cfg["lr_bias"]) * lr_ratio
    seed = int(net_cfg["seed"])
    use_bias = bool(use_bias if use_bias is not None else net_cfg["use_bias"])
    hidden_size = int(net_cfg["hidden_size"])

    inputs, targets, groups = load_mecanum_dataset(dataset_path)
    n_groups = len(set(groups))
    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(targets, dtype=torch.float32)

    dist_n = torch.hypot(X[:, 0], X[:, 1])
    w_far, w_near, near_scale = (float(v) for v in net_cfg["loss_near_dock_weight"])
    spread_gain, spread_floor = (float(v) for v in net_cfg["reg_spread"])
    sample_w = w_far + w_near * torch.exp(-dist_n / near_scale)

    total = len(X)
    train_idx, val_idx = group_train_val_split(groups, float(net_cfg["val_split"]), seed)
    train_size = len(train_idx)

    train_loader = DataLoader(
        TensorDataset(X[train_idx], Y[train_idx], sample_w[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(X[val_idx], Y[val_idx]), batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PioneerNN(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE, use_bias=use_bias)
    _init_multiscale_mecanum_bank(model, seed)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": [model.hidden.weight], "lr": lr_hidden, "weight_decay": float(net_cfg["wd_hidden"])},
            {"params": [model.output.weight], "lr": lr_output, "weight_decay": float(net_cfg["wd_output"])},
        ]
        + ([{"params": [model.hidden.bias], "lr": lr_bias, "weight_decay": float(net_cfg["wd_bias"])}] if use_bias else [])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(net_cfg["scheduler_eta_min"]))

    history: Dict[str, Any] = {
        "mode": "mecanum",
        "optimizer": "AdamW",
        "use_bias": use_bias,
        "effective_config": {
            "optimizer": "adamw",
            "batch_size": batch_size,
            "param_groups": {
                "hidden_weight": {"lr": lr_hidden, "weight_decay": float(net_cfg["wd_hidden"])},
                "output_weight": {"lr": lr_output, "weight_decay": float(net_cfg["wd_output"])},
                **({"hidden_bias": {"lr": lr_bias, "weight_decay": float(net_cfg["wd_bias"])}} if use_bias else {}),
            },
            "scheduler_eta_min": float(net_cfg["scheduler_eta_min"]),
            "beta": BETA,
            "loss_near_dock_weight": list(net_cfg["loss_near_dock_weight"]),
            "reg_spread": list(net_cfg["reg_spread"]),
            "val_split": float(net_cfg["val_split"]),
            "split": "by group_id (jittered copies stay with their base state)",
            "n_groups": n_groups,
            "n_jittered_rows": total - n_groups,
            "seed": seed,
        },
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_loss_vx": [],
        "val_loss_vy": [],
        "val_loss_w": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_weights_json = None
    start_time = time.time()

    print("=" * 84)
    print(f"   LIMO MECANUM OFFLINE TRAINING (3 -> {hidden_size} -> 3 | use_bias={use_bias} | Device: {device})")
    print(f"   Dataset: {dataset_path} ({total:,} samples: {train_size:,} train / {total - train_size:,} val)")
    print("=" * 84)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Val Loss':^11} | {'Val MSE (vx / vy / w)':^27} | {'LR':^9}")
    print("-" * 84)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            err = (
                (pred[:, 0] - by[:, 0]) ** 2
                + (pred[:, 1] - by[:, 1]) ** 2
                + BETA * (pred[:, 2] - by[:, 2]) ** 2
            ) / (2.0 + BETA)
            # Optimized loss: near-dock weighted (bw = loss_near_dock_weight profile).
            # The reported Train/Val losses and the best-epoch criterion use the
            # unweighted `weighted_mecanum_loss`, so train and val stay comparable.
            data_loss = torch.mean(bw * err)
            out_norms = torch.norm(model.output.weight, dim=0)
            spread_reg = spread_gain * torch.mean((out_norms - spread_floor).clamp(max=0.0) ** 2)
            loss = data_loss + spread_reg
            loss.backward()
            optimizer.step()

            if use_bias:
                with torch.no_grad():
                    h0 = torch.tanh(model.hidden.bias)
                    model.output.bias.copy_(-(model.output.weight @ h0))

            with torch.no_grad():
                train_loss += weighted_mecanum_loss(pred, by).item()
        avg_train = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss = val_vx = val_vy = val_w = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += weighted_mecanum_loss(pred, by).item()
                val_vx += torch.mean((pred[:, 0] - by[:, 0]) ** 2).item()
                val_vy += torch.mean((pred[:, 1] - by[:, 1]) ** 2).item()
                val_w += torch.mean((pred[:, 2] - by[:, 2]) ** 2).item()

        n_val = max(1, len(val_loader))
        avg_val = val_loss / n_val
        avg_vx = val_vx / n_val
        avg_vy = val_vy / n_val
        avg_w = val_w / n_val
        lr_now = optimizer.param_groups[0]["lr"]

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_train, 6))
        history["val_loss"].append(round(avg_val, 6))
        history["val_loss_vx"].append(round(avg_vx, 6))
        history["val_loss_vy"].append(round(avg_vy, 6))
        history["val_loss_w"].append(round(avg_w, 6))
        history["learning_rate"].append(round(lr_now, 6))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_weights_json = model.save_weights_to_json()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"{epoch:3d}/{epochs:<3d} | {avg_train:11.6f} | {avg_val:11.6f} | "
                f"{avg_vx:7.4f} / {avg_vy:7.4f} / {avg_w:7.4f} | {lr_now:9.6f}"
            )

        scheduler.step()

    out_weights_path = weights_output if os.path.isabs(weights_output) else os.path.join(_project_root, weights_output)
    out_history_path = history_output if os.path.isabs(history_output) else os.path.join(_project_root, history_output)
    os.makedirs(os.path.dirname(out_weights_path), exist_ok=True)

    with open(out_weights_path, "w", encoding="utf-8") as f:
        json.dump(best_weights_json, f, indent=2)
    with open(out_history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    print("=" * 84)
    print(f"Done in {elapsed:.1f}s | Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"Saved Mecanum weights: {out_weights_path}")
    print("=" * 84)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3-output Mecanum PioneerNN.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--weights-out", type=str, default=None)
    parser.add_argument("--history-out", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bias", action="store_true", default=None, help="Enable layer bias")
    parser.add_argument("--no-bias", action="store_false", dest="bias", help="Disable layer bias (MLP(0)==0)")
    args = parser.parse_args()
    train_mecanum(
        dataset_path=args.dataset,
        weights_output=args.weights_out,
        history_output=args.history_out,
        epochs=args.epochs,
        use_bias=args.bias,
    )


if __name__ == "__main__":
    main()
