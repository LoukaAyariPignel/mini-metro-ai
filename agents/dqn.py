"""Agent Deep Q-Network (Double DQN + masque d'actions).

L'agent apprend, par essai-erreur, quelle action de construction (étendre
telle ligne vers telle station, effacer une ligne, ne rien faire) maximise
le nombre de passagers livrés sur la durée d'une partie.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from mini_metro.env import N_ACTIONS, OBS_SIZE

NEG_INF = -1e9


class QNetwork(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.995,
        batch_size: int = 128,
        target_sync_every: int = 2000,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer()
        self.target_sync_every = target_sync_every
        self._updates = 0

    # ------------------------------------------------------------------ agir

    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return int(np.random.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            q = self.q(x).squeeze(0).cpu().numpy()
        q[~mask] = NEG_INF
        return int(np.argmax(q))

    # ------------------------------------------------------------- apprendre

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size * 4:
            return None
        batch = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(np.stack([t.obs for t in batch]), device=self.device)
        actions = torch.as_tensor([t.action for t in batch], device=self.device)
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_obs = torch.as_tensor(np.stack([t.next_obs for t in batch]), device=self.device)
        next_masks = torch.as_tensor(np.stack([t.next_mask for t in batch]), device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN : sélection par le réseau courant, évaluation par la cible
            next_q_online = self.q(next_obs).masked_fill(~next_masks, NEG_INF)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, best_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self._updates += 1
        if self._updates % self.target_sync_every == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # ------------------------------------------------------- sauvegarde/chargement

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state)
        self.target.load_state_dict(state)
