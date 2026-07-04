"""Environnement d'apprentissage par renforcement autour de MiniMetroGame.

API compatible Gymnasium : reset() -> (obs, info), step(a) -> (obs, reward,
terminated, truncated, info), plus action_mask() pour les actions valides.

Espace d'actions (discret) :
    0                                : ne rien faire
    1 .. MAX_LINES                   : effacer la ligne l
    MAX_LINES+1 ..                   : étendre la ligne l vers la station s
                                       (index = 1 + MAX_LINES + l*MAX_STATIONS + s)

Récompense :
    +1.0  par passager livré
    +0.3  la première fois qu'une station est reliée à une ligne
    -0.02 par station en surcharge à chaque pas
    -0.1  pour un effacement de ligne (décourage les resets frénétiques)
    -20.0 en cas de game over
"""

from __future__ import annotations

import numpy as np

from mini_metro.game import (
    MAX_LINE_LEN,
    MAX_LINES,
    MAX_STATIONS,
    N_SHAPES,
    OVERCROWD_LIMIT,
    STATION_CAPACITY,
    TRAIN_CAPACITY,
    MiniMetroGame,
)

DT = 1.0  # un pas d'environnement = 1 seconde de jeu

# Observation par station : présence, x, y, forme one-hot, file, timer surcharge
STATION_FEATS = 3 + N_SHAPES + 2
OBS_SIZE = (
    MAX_STATIONS * STATION_FEATS  # stations
    + MAX_LINES * MAX_STATIONS    # appartenance station-ligne
    + MAX_LINES * 3               # longueur, nb trains, charge des trains
    + 4                           # temps, nb stations, réserve de trains, attente totale
)

N_ACTIONS = 1 + MAX_LINES + MAX_LINES * MAX_STATIONS


class MiniMetroEnv:
    def __init__(self, seed: int | None = None, max_steps: int = 1800) -> None:
        self._seed = seed
        self.max_steps = max_steps
        self.game: MiniMetroGame | None = None
        self.steps = 0
        self._connected: set[int] = set()

    # ------------------------------------------------------------------- API

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._seed = seed
        self.game = MiniMetroGame(self._seed)
        if self._seed is not None:
            self._seed += 1  # épisode suivant différent mais reproductible
        self.steps = 0
        self._connected = set()
        return self._obs(), {}

    def step(self, action: int):
        assert self.game is not None, "reset() d'abord"
        g = self.game
        reward = 0.0

        if action == 0:
            pass
        elif action <= MAX_LINES:
            if g.clear_line(action - 1):
                reward -= 0.1
        else:
            a = action - 1 - MAX_LINES
            line_idx, station_idx = divmod(a, MAX_STATIONS)
            if g.extend_line(line_idx, station_idx) and station_idx not in self._connected:
                self._connected.add(station_idx)
                reward += 0.3

        events = g.tick(DT)
        self.steps += 1

        reward += events["delivered"] * 1.0
        reward -= events["overcrowded"] * 0.02

        terminated = g.game_over
        truncated = self.steps >= self.max_steps and not terminated
        if terminated:
            reward -= 20.0

        info = {"score": g.score, "time": g.time, **events}
        return self._obs(), reward, terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        """Booléens : quelles actions sont valides dans l'état courant."""
        g = self.game
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[0] = True
        if g is None or g.game_over:
            return mask
        for li in range(MAX_LINES):
            if g.can_clear(li):
                mask[1 + li] = True
            for si in range(len(g.stations)):
                if g.can_extend(li, si):
                    mask[1 + MAX_LINES + li * MAX_STATIONS + si] = True
        return mask

    # ------------------------------------------------------------ observation

    def _obs(self) -> np.ndarray:
        g = self.game
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        o = 0
        for i in range(MAX_STATIONS):
            if i < len(g.stations):
                st = g.stations[i]
                obs[o] = 1.0
                obs[o + 1] = st.x
                obs[o + 2] = st.y
                obs[o + 3 + int(st.shape)] = 1.0
                obs[o + 3 + N_SHAPES] = min(len(st.queue) / STATION_CAPACITY, 2.0)
                obs[o + 4 + N_SHAPES] = st.overcrowd_timer / OVERCROWD_LIMIT
            o += STATION_FEATS
        for line in g.lines:
            for si in line.stations:
                obs[o + si] = 1.0
            o += MAX_STATIONS
        for li, line in enumerate(g.lines):
            trains = [tr for tr in g.trains if tr.line_idx == li]
            load = sum(len(tr.passengers) for tr in trains)
            obs[o] = len(line) / MAX_LINE_LEN
            obs[o + 1] = len(trains) / 3.0
            obs[o + 2] = load / (TRAIN_CAPACITY * 3.0)
            o += 3
        obs[o] = min(g.time / 1800.0, 2.0)
        obs[o + 1] = len(g.stations) / MAX_STATIONS
        obs[o + 2] = g.trains_pool / 6.0
        obs[o + 3] = min(g.total_waiting() / 40.0, 2.0)
        return obs
