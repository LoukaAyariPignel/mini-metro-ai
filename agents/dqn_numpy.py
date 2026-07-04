"""Inférence du DQN en pur numpy — pour tourner sur le téléphone (Termux),
où PyTorch n'est pas disponible.

Les poids sont exportés depuis le modèle PyTorch par export_weights.py
(models/dqn_weights.npz). Même API act(obs, mask) que DQNAgent.
"""

from __future__ import annotations

import numpy as np

NEG_INF = -1e9


class NumpyDQN:
    def __init__(self, weights_path: str = "models/dqn_weights.npz") -> None:
        data = np.load(weights_path)
        # couches Linear successives : w0/b0, w1/b1, w2/b2
        self.layers = []
        i = 0
        while f"w{i}" in data:
            self.layers.append((data[f"w{i}"], data[f"b{i}"]))
            i += 1
        if not self.layers:
            raise ValueError(f"Aucun poids trouvé dans {weights_path}")

    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        x = obs.astype(np.float32)
        for k, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if k < len(self.layers) - 1:
                x = np.maximum(x, 0.0)  # ReLU
        x[~mask] = NEG_INF
        return int(np.argmax(x))
