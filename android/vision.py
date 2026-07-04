"""Vision : extraire l'état du jeu depuis une capture d'écran de Mini Metro.

Le style graphique minimaliste du jeu rend l'analyse fiable :
- stations = grands anneaux sombres (cercle, triangle, carré, étoile, pentagone)
- passagers = petites formes sombres pleines à côté des stations
- lignes    = traits épais très colorés (rouge, bleu, jaune, ...)

Tous les seuils sont regroupés dans VisionConfig pour pouvoir être calibrés
sur de vraies captures (voir run_android.py --dry-run qui produit des images
annotées).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from mini_metro.game import Shape


@dataclass
class DetectedStation:
    x: float            # pixels
    y: float
    shape: Shape
    radius: float
    queue_len: int = 0


@dataclass
class DetectedLine:
    color_name: str
    stations: list[int] = field(default_factory=list)  # indices ordonnés


@dataclass
class VisionState:
    stations: list[DetectedStation]
    lines: list[DetectedLine]
    width: int
    height: int
    game_over: bool = False


@dataclass
class VisionConfig:
    # zone de jeu : fractions de l'écran à ignorer (UI en haut / en bas)
    crop_top: float = 0.05
    crop_bottom: float = 0.08
    # seuil "sombre" pour stations et passagers (sur image en niveaux de gris)
    dark_threshold: int = 110
    # saturation max des éléments "sombres" : les symboles sont noirs, les
    # lignes de métro sont très colorées — on les exclut du masque sombre
    dark_max_saturation: int = 80
    # aire des symboles de station, en fraction de l'aire de l'écran
    station_area_min: float = 4e-4
    station_area_max: float = 4e-3
    # aire des passagers
    passenger_area_min: float = 2e-5
    passenger_area_max: float = 3.5e-4
    # rayon (multiples du rayon de la station) où chercher les passagers
    passenger_search_radius: float = 3.2
    # couleurs des lignes en HSV (min, max) — palette classique de Mini Metro
    line_colors: dict = field(default_factory=lambda: {
        "jaune": ((20, 90, 120), (35, 255, 255)),
        "rouge": ((0, 90, 120), (9, 255, 255)),
        "rouge2": ((170, 90, 120), (180, 255, 255)),   # le rouge chevauche 0°
        "bleu": ((95, 90, 120), (125, 255, 255)),
    })
    # une station appartient à une ligne si des pixels de sa couleur sont
    # présents à moins de ce rayon (multiples du rayon station)
    line_membership_radius: float = 1.8
    # luminosité moyenne sous laquelle on considère l'écran "game over" (voile sombre)
    game_over_brightness: float = 90.0


# ---------------------------------------------------------------------- formes

def _classify_shape(contour: np.ndarray) -> Shape:
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if peri <= 0 or area <= 0:
        return Shape.CIRCLE
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    n = len(approx)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1.0
    circularity = 4.0 * np.pi * area / (peri * peri)

    if solidity < 0.75:          # forte concavité → étoile
        return Shape.STAR
    if n == 3:
        return Shape.TRIANGLE
    if n == 4:
        return Shape.SQUARE
    if n == 5 and circularity < 0.83:
        return Shape.PENTAGON
    return Shape.CIRCLE


# ---------------------------------------------------------------------- parsing

def parse_screenshot(img_bgr: np.ndarray, cfg: VisionConfig | None = None) -> VisionState:
    cfg = cfg or VisionConfig()
    h, w = img_bgr.shape[:2]
    y0, y1 = int(h * cfg.crop_top), int(h * (1.0 - cfg.crop_bottom))
    field_img = img_bgr[y0:y1]
    gray = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)

    game_over = float(gray.mean()) < cfg.game_over_brightness

    saturation = cv2.cvtColor(field_img, cv2.COLOR_BGR2HSV)[:, :, 1]
    dark = (
        (gray < cfg.dark_threshold) & (saturation < cfg.dark_max_saturation)
    ).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    screen_area = float(w * h)
    stations: list[DetectedStation] = []
    passengers: list[tuple[float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        frac = area / screen_area
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"] + y0
        if cfg.station_area_min <= frac <= cfg.station_area_max:
            radius = float(np.sqrt(area / np.pi))
            stations.append(DetectedStation(cx, cy, _classify_shape(c), radius))
        elif cfg.passenger_area_min <= frac < cfg.station_area_min:
            passengers.append((cx, cy))  # cy inclut déjà le décalage y0

    # rattache chaque passager à la station la plus proche
    for px, py in passengers:
        best, best_d = None, float("inf")
        for st in stations:
            d = float(np.hypot(st.x - px, st.y - py))
            if d < best_d:
                best, best_d = st, d
        if best is not None and best_d <= best.radius * cfg.passenger_search_radius:
            best.queue_len += 1

    lines = _detect_lines(field_img, y0, stations, cfg)
    return VisionState(stations, lines, w, h, game_over)


def _detect_lines(
    field_img: np.ndarray,
    y0: int,
    stations: list[DetectedStation],
    cfg: VisionConfig,
) -> list[DetectedLine]:
    hsv = cv2.cvtColor(field_img, cv2.COLOR_BGR2HSV)
    masks: dict[str, np.ndarray] = {}
    for name, (lo, hi) in cfg.line_colors.items():
        base = name.rstrip("2")  # "rouge2" complète "rouge"
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        masks[base] = cv2.bitwise_or(masks[base], m) if base in masks else m

    lines: list[DetectedLine] = []
    for name, mask in masks.items():
        if int(mask.sum()) < 255 * 50:  # moins de ~50 pixels : pas de ligne
            continue
        members = []
        for i, st in enumerate(stations):
            r = int(st.radius * cfg.line_membership_radius)
            x_lo, x_hi = int(st.x - r), int(st.x + r)
            y_lo, y_hi = int(st.y - y0 - r), int(st.y - y0 + r)
            patch = mask[max(y_lo, 0):max(y_hi, 0), max(x_lo, 0):max(x_hi, 0)]
            if patch.size and int((patch > 0).sum()) > 8:
                members.append(i)
        if len(members) >= 2:
            ordered = _order_line_stations(members, stations, mask, y0)
            lines.append(DetectedLine(name, ordered))
    return lines


def _segment_covered(
    mask: np.ndarray, a: DetectedStation, b: DetectedStation, y0: int
) -> float:
    """Fraction du segment [a,b] couverte par des pixels de la couleur."""
    n = 24
    hits = 0
    h, w = mask.shape
    for k in range(1, n):
        t = k / n
        x = int(a.x + (b.x - a.x) * t)
        y = int(a.y + (b.y - a.y) * t) - y0
        if 0 <= x < w and 0 <= y < h:
            r = 6
            if mask[max(y - r, 0):y + r, max(x - r, 0):x + r].any():
                hits += 1
    return hits / (n - 1)


def _order_line_stations(
    members: list[int],
    stations: list[DetectedStation],
    mask: np.ndarray,
    y0: int,
) -> list[int]:
    """Ordonne les stations d'une ligne en suivant les segments colorés."""
    # graphe : deux stations sont adjacentes si le segment entre elles est coloré
    adj: dict[int, list[int]] = {i: [] for i in members}
    for ii, i in enumerate(members):
        for j in members[ii + 1:]:
            if _segment_covered(mask, stations[i], stations[j], y0) > 0.75:
                adj[i].append(j)
                adj[j].append(i)
    # on part d'une extrémité (degré 1), sinon de la première station
    start = next((i for i in members if len(adj[i]) == 1), members[0])
    ordered, seen = [start], {start}
    while True:
        nxt = [j for j in adj[ordered[-1]] if j not in seen]
        if not nxt:
            break
        # préfère le voisin le plus proche (évite de sauter une station alignée)
        nxt.sort(key=lambda j: np.hypot(
            stations[j].x - stations[ordered[-1]].x,
            stations[j].y - stations[ordered[-1]].y,
        ))
        ordered.append(nxt[0])
        seen.add(nxt[0])
    # stations membres non reliées par le graphe : ajoutées en bout par proximité
    for i in members:
        if i not in seen:
            ordered.append(i)
            seen.add(i)
    return ordered


# -------------------------------------------------------------------- annotation

SHAPE_NAMES = {
    Shape.CIRCLE: "cercle",
    Shape.TRIANGLE: "triangle",
    Shape.SQUARE: "carre",
    Shape.STAR: "etoile",
    Shape.PENTAGON: "pentagone",
}


def annotate(img_bgr: np.ndarray, state: VisionState) -> np.ndarray:
    """Dessine les détections sur l'image (pour calibrer)."""
    out = img_bgr.copy()
    for i, st in enumerate(state.stations):
        p = (int(st.x), int(st.y))
        cv2.circle(out, p, int(st.radius * 1.6), (0, 200, 0), 2)
        cv2.putText(
            out,
            f"{i}:{SHAPE_NAMES[st.shape]} q={st.queue_len}",
            (p[0] - 40, p[1] - int(st.radius * 1.8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
        )
    for line in state.lines:
        pts = [state.stations[i] for i in line.stations]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, (int(a.x), int(a.y)), (int(b.x), int(b.y)), (255, 0, 255), 2)
        if pts:
            cv2.putText(
                out, line.color_name,
                (int(pts[0].x) + 10, int(pts[0].y) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2,
            )
    if state.game_over:
        cv2.putText(out, "GAME OVER ?", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return out
