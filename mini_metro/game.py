"""Simulation simplifiée mais fidèle de Mini Metro.

Règles reproduites :
- Des stations apparaissent progressivement sur la carte, chacune avec une
  forme (cercle, triangle, carré + formes rares étoile/pentagone).
- Des passagers apparaissent dans les stations ; chaque passager veut
  rejoindre une station de la forme qu'il porte.
- Le joueur trace des lignes de métro (listes ordonnées de stations).
  Chaque ligne fait circuler un ou plusieurs trains en aller-retour.
- Un train charge les passagers dont la destination est desservie par sa
  ligne, et les dépose dès qu'il atteint une station de la bonne forme.
- Si une station déborde (file > capacité) trop longtemps : partie perdue.
- Chaque semaine de jeu, un train supplémentaire est accordé.

Le score est le nombre de passagers livrés.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum


class Shape(IntEnum):
    CIRCLE = 0
    TRIANGLE = 1
    SQUARE = 2
    STAR = 3
    PENTAGON = 4


N_SHAPES = 5

# Paramètres du jeu (unités : la carte est un carré [0,1]², le temps en secondes)
MAX_STATIONS = 16
MAX_LINES = 3
MAX_LINE_LEN = 10
STATION_CAPACITY = 6
TRAIN_CAPACITY = 6
TRAIN_SPEED = 0.14          # unités de carte / seconde
DWELL_TIME = 1.0            # arrêt en station (s)
OVERCROWD_LIMIT = 30.0      # secondes de débordement avant game over
STATION_SPAWN_EVERY = 45.0  # nouvelle station toutes les 45 s
WEEK_LENGTH = 210.0         # une "semaine" de jeu
MIN_STATION_DIST = 0.13

# Distribution des formes des nouvelles stations
SHAPE_WEIGHTS = {
    Shape.CIRCLE: 0.45,
    Shape.TRIANGLE: 0.25,
    Shape.SQUARE: 0.16,
    Shape.STAR: 0.08,
    Shape.PENTAGON: 0.06,
}


@dataclass
class Station:
    idx: int
    x: float
    y: float
    shape: Shape
    queue: list[Shape] = field(default_factory=list)  # destinations des passagers en attente
    overcrowd_timer: float = 0.0


@dataclass
class Train:
    line_idx: int
    edge: int = 0          # index du segment courant sur la ligne
    t: float = 0.0         # progression [0,1] sur le segment
    direction: int = 1     # +1 aller, -1 retour
    dwell: float = 0.0     # temps d'arrêt restant
    passengers: list[Shape] = field(default_factory=list)


class Line:
    def __init__(self) -> None:
        self.stations: list[int] = []

    def __len__(self) -> int:
        return len(self.stations)


class MiniMetroGame:
    """Moteur de jeu. Avancer d'un pas avec tick(dt)."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.time = 0.0
        self.score = 0
        self.game_over = False
        self.stations: list[Station] = []
        self.lines = [Line() for _ in range(MAX_LINES)]
        self.trains: list[Train] = []
        self.trains_pool = MAX_LINES + 1   # trains disponibles au départ
        self._next_station_at = 0.0
        self._next_week_at = WEEK_LENGTH
        # état dérivé, mis à jour à chaque changement de ligne
        self._shapes_on_line: list[set[Shape]] = [set() for _ in range(MAX_LINES)]

        for _ in range(3):
            self._spawn_station(initial=True)
        self._next_station_at = STATION_SPAWN_EVERY

    # ------------------------------------------------------------------ carte

    def _spawn_station(self, initial: bool = False) -> None:
        if len(self.stations) >= MAX_STATIONS:
            return
        for _ in range(200):
            x = self.rng.uniform(0.08, 0.92)
            y = self.rng.uniform(0.08, 0.92)
            if all(math.hypot(s.x - x, s.y - y) >= MIN_STATION_DIST for s in self.stations):
                break
        else:
            return
        if initial:
            # les 3 premières stations couvrent les 3 formes de base
            shape = [Shape.CIRCLE, Shape.TRIANGLE, Shape.SQUARE][len(self.stations)]
        else:
            shapes = list(SHAPE_WEIGHTS)
            weights = list(SHAPE_WEIGHTS.values())
            # les formes rares n'apparaissent qu'après la première semaine
            if self.time < WEEK_LENGTH:
                shapes, weights = shapes[:3], weights[:3]
            shape = self.rng.choices(shapes, weights)[0]
        self.stations.append(Station(len(self.stations), x, y, shape))

    def _existing_shapes(self) -> set[Shape]:
        return {s.shape for s in self.stations}

    # ---------------------------------------------------------------- actions

    def can_extend(self, line_idx: int, station_idx: int) -> bool:
        if self.game_over or station_idx >= len(self.stations):
            return False
        line = self.lines[line_idx]
        return station_idx not in line.stations and len(line) < MAX_LINE_LEN

    def extend_line(self, line_idx: int, station_idx: int) -> bool:
        """Ajoute une station à l'extrémité de la ligne la plus proche."""
        if not self.can_extend(line_idx, station_idx):
            return False
        line = self.lines[line_idx]
        st = self.stations[station_idx]
        if len(line) < 2:
            line.stations.append(station_idx)
        else:
            head = self.stations[line.stations[0]]
            tail = self.stations[line.stations[-1]]
            d_head = math.hypot(head.x - st.x, head.y - st.y)
            d_tail = math.hypot(tail.x - st.x, tail.y - st.y)
            if d_head < d_tail:
                line.stations.insert(0, station_idx)
                # décale les trains de cette ligne (un segment ajouté en tête)
                for tr in self.trains:
                    if tr.line_idx == line_idx:
                        tr.edge += 1
            else:
                line.stations.append(station_idx)
        self._refresh_line_shapes(line_idx)
        self._maybe_add_train(line_idx)
        return True

    def can_clear(self, line_idx: int) -> bool:
        return not self.game_over and len(self.lines[line_idx]) > 0

    def clear_line(self, line_idx: int) -> bool:
        """Supprime la ligne ; ses trains retournent dans la réserve."""
        if not self.can_clear(line_idx):
            return False
        removed = [tr for tr in self.trains if tr.line_idx == line_idx]
        for tr in removed:
            # les passagers à bord retournent à la station la plus proche
            station = self.stations[self._nearest_station_on_line(tr)]
            station.queue.extend(tr.passengers)
        self.trains = [tr for tr in self.trains if tr.line_idx != line_idx]
        self.trains_pool += len(removed)
        self.lines[line_idx] = Line()
        self._refresh_line_shapes(line_idx)
        return True

    def _nearest_station_on_line(self, tr: Train) -> int:
        line = self.lines[tr.line_idx]
        i = min(tr.edge + (1 if tr.t > 0.5 else 0), len(line) - 1)
        return line.stations[i]

    def _refresh_line_shapes(self, line_idx: int) -> None:
        self._shapes_on_line[line_idx] = {
            self.stations[i].shape for i in self.lines[line_idx].stations
        }

    def _maybe_add_train(self, line_idx: int) -> None:
        has_train = any(tr.line_idx == line_idx for tr in self.trains)
        if len(self.lines[line_idx]) >= 2 and not has_train and self.trains_pool > 0:
            self.trains_pool -= 1
            self.trains.append(Train(line_idx))

    def assign_extra_train(self) -> None:
        """Affecte un train de réserve à la ligne la plus chargée."""
        if self.trains_pool <= 0:
            return
        best, best_load = None, -1
        for li, line in enumerate(self.lines):
            if len(line) < 2:
                continue
            load = sum(len(self.stations[s].queue) for s in line.stations)
            n_trains = sum(1 for tr in self.trains if tr.line_idx == li)
            if n_trains >= 3:
                continue
            if load > best_load:
                best, best_load = li, load
        if best is not None:
            self.trains_pool -= 1
            # démarre à l'autre extrémité pour répartir la desserte
            self.trains.append(
                Train(best, edge=max(len(self.lines[best]) - 2, 0), direction=-1)
            )

    # ------------------------------------------------------------- simulation

    def tick(self, dt: float) -> dict:
        """Avance la simulation de dt secondes. Retourne les événements du pas."""
        events = {"delivered": 0, "picked_up": 0, "overcrowded": 0}
        if self.game_over:
            return events
        self.time += dt

        if self.time >= self._next_station_at:
            self._spawn_station()
            self._next_station_at += STATION_SPAWN_EVERY
        if self.time >= self._next_week_at:
            self.trains_pool += 1
            self.assign_extra_train()
            self._next_week_at += WEEK_LENGTH

        self._spawn_passengers(dt)
        for tr in self.trains:
            self._move_train(tr, dt, events)

        for st in self.stations:
            if len(st.queue) > STATION_CAPACITY:
                st.overcrowd_timer += dt
                events["overcrowded"] += 1
                if st.overcrowd_timer >= OVERCROWD_LIMIT:
                    self.game_over = True
            else:
                st.overcrowd_timer = max(0.0, st.overcrowd_timer - dt * 0.5)
        return events

    def _spawn_passengers(self, dt: float) -> None:
        # le rythme augmente avec le temps : la difficulté monte comme dans le jeu
        rate = 0.018 + self.time * 1.1e-5  # passagers / s / station
        shapes_present = self._existing_shapes()
        for st in self.stations:
            if self.rng.random() < rate * dt:
                choices = [s for s in shapes_present if s != st.shape]
                if choices:
                    st.queue.append(self.rng.choice(choices))

    def _move_train(self, tr: Train, dt: float, events: dict) -> None:
        line = self.lines[tr.line_idx]
        if len(line) < 2:
            return
        if tr.dwell > 0:
            tr.dwell -= dt
            return
        a = self.stations[line.stations[tr.edge]]
        b = self.stations[line.stations[tr.edge + 1]]
        seg = math.hypot(b.x - a.x, b.y - a.y)
        tr.t += (TRAIN_SPEED * dt / max(seg, 1e-6)) * tr.direction
        arrived: Station | None = None
        if tr.t >= 1.0:
            arrived = b
            if tr.edge + 1 >= len(line) - 1:
                tr.direction = -1
                tr.edge = len(line) - 2
                tr.t = 1.0
            else:
                tr.edge += 1
                tr.t = 0.0
        elif tr.t <= 0.0:
            arrived = a
            if tr.edge <= 0:
                tr.direction = 1
                tr.edge = 0
                tr.t = 0.0
            else:
                tr.edge -= 1
                tr.t = 1.0
        if arrived is not None:
            self._serve_station(tr, arrived, events)
            tr.dwell = DWELL_TIME

    def _serve_station(self, tr: Train, st: Station, events: dict) -> None:
        # dépose
        before = len(tr.passengers)
        tr.passengers = [p for p in tr.passengers if p != st.shape]
        delivered = before - len(tr.passengers)
        self.score += delivered
        events["delivered"] += delivered
        # embarque les passagers dont la destination est desservie par la ligne
        served = self._shapes_on_line[tr.line_idx]
        remaining: list[Shape] = []
        for p in st.queue:
            if len(tr.passengers) < TRAIN_CAPACITY and p in served and p != st.shape:
                tr.passengers.append(p)
                events["picked_up"] += 1
            else:
                remaining.append(p)
        st.queue = remaining

    # ---------------------------------------------------------------- helpers

    def station_line_membership(self, station_idx: int) -> list[bool]:
        return [station_idx in line.stations for line in self.lines]

    def total_waiting(self) -> int:
        return sum(len(s.queue) for s in self.stations)
