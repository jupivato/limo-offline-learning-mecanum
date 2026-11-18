"""
Offline Behavioral Cloning Training Pipeline for AgileX LIMO (Mecanum 3-Output)
================================================================================
Trains a separate 3-output PioneerNN controller (vx, vy, w) on
data/dataset_mecanum.csv and saves the weights to models/pioneer_mecanum_weights.json
without modifying any existing differential-drive models or scripts.

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
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.pioneer_nn import PioneerNN, recast_angle

# Hyperparameters matching canonical PioneerNN setup
INPUT_SIZE = 3
HIDDEN_SIZE = 1000
OUTPUT_SIZE = 3
USE_BIAS = True

LEARNING_RATE = 0.005
EPOCHS = 60
BATCH_SIZE = 64
BETA = 0.3
WEIGHT_DECAY = 1e-4

V_MAX = 0.50
W_MAX = 1.00
X_MAX = 5.0
Y_MAX = 5.0

DEFAULT_DATASET = "data/dataset_mecanum.csv"
DEFAULT_WEIGHTS = "models/pioneer_mecanum_weights.json"
DEFAULT_HISTORY = "models/training_history_mecanum.json"


def load_mecanum_dataset(csv_path: str) -> Tuple[List[List[float]], List[List[float]]]:
    """Loads 6-column Mecanum dataset and normalizes inputs and 3 outputs to [-1, 1]."""
    resolved = csv_path if os.path.exists(csv_path) else os.path.join(_project_root, csv_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Mecanum dataset not found: {csv_path}")

    inputs, targets = [], []
    with open(resolved, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xr = float(row["x_target_robot"])
                yr = float(row["y_target_robot"])
                thr = recast_angle(float(row["theta_robot"]))
                vx = float(row["vx_robot"])
                vy = float(row["vy_robot"])
                w = float(row["w_robot"])

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
            except (KeyError, ValueError):
                continue

    if not inputs:
        raise ValueError(f"No valid samples found in {resolved}")
    return inputs, targets


def weighted_mecanum_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = BETA) -> torch.Tensor:
    """Weighted MSE across (vx, vy, w): (MSE(vx) + MSE(vy) + beta * MSE(w)) / (2 + beta)."""
    loss_vx = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
    loss_vy = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
    loss_w = torch.mean((pred[:, 2] - target[:, 2]) ** 2)
    return (loss_vx + loss_vy + beta * loss_w) / (2.0 + beta)


def train_mecanum(
    dataset_path: str = DEFAULT_DATASET,
    weights_output: str = DEFAULT_WEIGHTS,
    history_output: str = DEFAULT_HISTORY,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> Dict[str, Any]:
    """Trains the 3-output Mecanum PioneerNN and saves best validation weights."""
    inputs, targets = load_mecanum_dataset(dataset_path)
    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(targets, dtype=torch.float32)

    total = len(X)
    train_size = int(0.80 * total)
    torch.manual_seed(42)
    perm = torch.randperm(total)
    train_idx, val_idx = perm[:train_size], perm[train_size:]

    train_loader = DataLoader(TensorDataset(X[train_idx], Y[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[val_idx], Y[val_idx]), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PioneerNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, use_bias=USE_BIAS)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: Dict[str, Any] = {
        "mode": "mecanum",
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
    print(f"   LIMO MECANUM OFFLINE TRAINING (3 -> {HIDDEN_SIZE} -> 3 | Device: {device})")
    print(f"   Dataset: {dataset_path} ({total:,} samples: {train_size:,} train / {total - train_size:,} val)")
    print("=" * 84)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Val Loss':^11} | {'Val MSE (vx / vy / w)':^27} | {'LR':^9}")
    print("-" * 84)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = weighted_mecanum_loss(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
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
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--weights-out", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--history-out", type=str, default=DEFAULT_HISTORY)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train_mecanum(
        dataset_path=args.dataset,
        weights_output=args.weights_out,
        history_output=args.history_out,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
