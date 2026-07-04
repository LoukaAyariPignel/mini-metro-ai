"""Agent aléatoire : baseline minimale (agit rarement, au hasard)."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, seed: int | None = None, act_prob: float = 0.05) -> None:
        self.rng = np.random.default_rng(seed)
        self.act_prob = act_prob

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        # la plupart du temps ne rien faire, sinon une action valide au hasard
        if self.rng.random() > self.act_prob:
            return 0
        valid = np.flatnonzero(mask)
        return int(self.rng.choice(valid))
