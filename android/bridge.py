"""Pont : état vu à l'écran → décision de l'agent → geste tactile.

- Reconstruit une vue du jeu compatible avec les agents (GreedyAgent lit
  stations/lignes, le DQN reçoit le même vecteur d'observation qu'en
  simulation).
- Traduit l'action « étendre la ligne l vers la station s » en un drag depuis
  l'extrémité de ligne la plus proche vers la station cible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from android.vision import VisionState
from mini_metro.env import (
    MAX_LINES,
    MAX_STATIONS,
    N_ACTIONS,
    OBS_SIZE,
    STATION_FEATS,
)
from mini_metro.game import (
    MAX_LINE_LEN,
    N_SHAPES,
    STATION_CAPACITY,
    Shape,
)


@dataclass
class _StationView:
    idx: int
    x: float           # normalisé [0,1]
    y: float
    shape: Shape
    queue: list        # longueur = nb de passagers (destinations inconnues)
    px: float          # pixels (pour les gestes)
    py: float


@dataclass
class _LineView:
    stations: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.stations)


class GameView:
    """Vue du jeu reconstruite depuis l'écran, même interface que MiniMetroGame
    pour ce qu'utilisent les agents."""

    def __init__(self, vs: VisionState) -> None:
        self.vs = vs
        self.stations: list[_StationView] = []
        for i, st in enumerate(vs.stations[:MAX_STATIONS]):
            self.stations.append(_StationView(
                idx=i,
                x=st.x / vs.width,
                y=st.y / vs.height,
                shape=st.shape,
                # destinations inconnues depuis l'écran : on remplit avec des
                # jokers "toute forme sauf la mienne" pour l'heuristique
                queue=[Shape((int(st.shape) + 1) % 3)] * st.queue_len,
                px=st.x,
                py=st.y,
            ))
        self.lines = [_LineView() for _ in range(MAX_LINES)]
        for li, dl in enumerate(vs.lines[:MAX_LINES]):
            self.lines[li].stations = [s for s in dl.stations if s < len(self.stations)]

    def can_extend(self, line_idx: int, station_idx: int) -> bool:
        if station_idx >= len(self.stations):
            return False
        line = self.lines[line_idx]
        return station_idx not in line.stations and len(line) < MAX_LINE_LEN


def action_mask(view: GameView) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[0] = True
    for li in range(MAX_LINES):
        # pas d'effacement de ligne sur le vrai jeu (geste complexe) : interdit
        for si in range(len(view.stations)):
            if view.can_extend(li, si):
                mask[1 + MAX_LINES + li * MAX_STATIONS + si] = True
    return mask


def observation(view: GameView, elapsed_s: float, trains_pool: int = 0) -> np.ndarray:
    """Vecteur d'observation au même format que MiniMetroEnv._obs()."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    o = 0
    for i in range(MAX_STATIONS):
        if i < len(view.stations):
            st = view.stations[i]
            obs[o] = 1.0
            obs[o + 1] = st.x
            obs[o + 2] = st.y
            obs[o + 3 + int(st.shape)] = 1.0
            obs[o + 3 + N_SHAPES] = min(len(st.queue) / STATION_CAPACITY, 2.0)
            obs[o + 4 + N_SHAPES] = 0.0  # timer de surcharge non observable
        o += STATION_FEATS
    for line in view.lines:
        for si in line.stations:
            obs[o + si] = 1.0
        o += MAX_STATIONS
    for line in view.lines:
        obs[o] = len(line) / MAX_LINE_LEN
        obs[o + 1] = (1.0 / 3.0) if len(line) >= 2 else 0.0
        obs[o + 2] = 0.0  # charge des trains non suivie
        o += 3
    obs[o] = min(elapsed_s / 1800.0, 2.0)
    obs[o + 1] = len(view.stations) / MAX_STATIONS
    obs[o + 2] = trains_pool / 6.0
    obs[o + 3] = min(sum(len(s.queue) for s in view.stations) / 40.0, 2.0)
    return obs


@dataclass
class Gesture:
    """Un drag en pixels écran."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str


class GestureMapper:
    """Traduit les actions de l'agent en gestes.

    Une ligne vide n'existe pas à l'écran : la première extension d'une ligne
    vide est mémorisée, et le drag n'est exécuté qu'à la seconde (station A →
    station B trace la nouvelle ligne).
    """

    def __init__(self) -> None:
        self.pending_start: dict[int, int] = {}

    def to_gesture(self, view: GameView, action: int) -> Gesture | None:
        if action <= MAX_LINES:  # no-op ou effacement (non mappé)
            return None
        a = action - 1 - MAX_LINES
        line_idx, station_idx = divmod(a, MAX_STATIONS)
        if station_idx >= len(view.stations):
            return None
        target = view.stations[station_idx]
        line = view.lines[line_idx]

        if len(line) == 0:
            pending = self.pending_start.get(line_idx)
            if pending is None or pending >= len(view.stations) or pending == station_idx:
                self.pending_start[line_idx] = station_idx
                return None
            src = view.stations[pending]
            del self.pending_start[line_idx]
            return Gesture(src.px, src.py, target.px, target.py,
                           f"nouvelle ligne {line_idx + 1} : {pending}→{station_idx}")

        head = view.stations[line.stations[0]]
        tail = view.stations[line.stations[-1]]
        d_head = math.hypot(head.px - target.px, head.py - target.py)
        d_tail = math.hypot(tail.px - target.px, tail.py - target.py)
        src = head if d_head < d_tail else tail
        return Gesture(src.px, src.py, target.px, target.py,
                       f"ligne {line_idx + 1} : {src.idx}→{station_idx}")
