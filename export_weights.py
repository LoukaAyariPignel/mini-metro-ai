"""Exporte le modèle PyTorch en poids numpy (.npz) pour le téléphone.

Usage : python3 export_weights.py [models/dqn.pt] [models/dqn_weights.npz]
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def export(src: str = "models/dqn.pt", dst: str = "models/dqn_weights.npz") -> None:
    state = torch.load(src, map_location="cpu")
    arrays = {}
    linear_idx = 0
    # state_dict d'un nn.Sequential : net.0.weight, net.0.bias, net.2.weight...
    for key in sorted(state, key=lambda k: int(k.split(".")[1])):
        if key.endswith(".weight"):
            arrays[f"w{linear_idx}"] = state[key].numpy().astype(np.float32)
        elif key.endswith(".bias"):
            arrays[f"b{linear_idx}"] = state[key].numpy().astype(np.float32)
            linear_idx += 1
    np.savez_compressed(dst, **arrays)
    print(f"{src} -> {dst} ({linear_idx} couches Linear)")


if __name__ == "__main__":
    args = sys.argv[1:]
    export(*args[:2])
