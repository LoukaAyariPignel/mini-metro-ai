"""Agent heuristique « glouton » : baseline solide, écrite à la main.

Stratégie :
1. Relier toute station orpheline à la ligne dont l'extrémité est la plus
   proche (en équilibrant la longueur des lignes).
2. S'assurer que chaque ligne dessert les formes demandées par ses passagers :
   si une ligne a des passagers en attente vers une forme qu'elle ne dessert
   pas, l'étendre vers la station de cette forme la plus proche.
3. Sinon, ne rien faire.
"""

from __future__ import annotations

import math

import numpy as np

from mini_metro.env import MAX_LINES, MAX_STATIONS, MiniMetroEnv


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


class GreedyAgent:
    def act_env(self, env: MiniMetroEnv) -> int:
        g = env.game
        connected = {s for line in g.lines for s in line.stations}

        # 1. stations orphelines (les plus surchargées d'abord)
        orphans = [s for s in g.stations if s.idx not in connected]
        orphans.sort(key=lambda s: -len(s.queue))
        for st in orphans:
            best, best_cost = None, float("inf")
            for li, line in enumerate(g.lines):
                if not g.can_extend(li, st.idx):
                    continue
                if len(line) == 0:
                    cost = 0.35  # ouvrir une ligne vide a un coût moyen
                else:
                    ends = [g.stations[line.stations[0]], g.stations[line.stations[-1]]]
                    cost = min(_dist(e, st) for e in ends) + 0.05 * len(line)
                if cost < best_cost:
                    best, best_cost = li, cost
            if best is not None:
                return 1 + MAX_LINES + best * MAX_STATIONS + st.idx

        # 2. formes demandées mais non desservies par la ligne
        for li, line in enumerate(g.lines):
            if len(line) < 2:
                continue
            served = {g.stations[s].shape for s in line.stations}
            wanted = {p for s in line.stations for p in g.stations[s].queue}
            missing = wanted - served
            if not missing:
                continue
            ends = [g.stations[line.stations[0]], g.stations[line.stations[-1]]]
            best, best_d = None, float("inf")
            for st in g.stations:
                if st.shape in missing and g.can_extend(li, st.idx):
                    d = min(_dist(e, st) for e in ends)
                    if d < best_d:
                        best, best_d = st.idx, d
            if best is not None:
                return 1 + MAX_LINES + li * MAX_STATIONS + best

        return 0

    # même signature que les autres agents (l'env est passé via closure dans
    # evaluate.py ; obs/mask ignorés car l'heuristique lit l'état du jeu)
    def act(self, obs: np.ndarray, mask: np.ndarray, env: MiniMetroEnv | None = None) -> int:
        assert env is not None
        action = self.act_env(env)
        return action if mask[action] else 0
