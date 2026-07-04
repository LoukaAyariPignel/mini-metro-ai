"""Faire jouer l'IA sur la vraie appli Android Mini Metro, via ADB (sans root).

Prérequis (une seule fois) :
  1. Sur le Pixel 7a : Paramètres > À propos du téléphone > taper 7 fois sur
     « Numéro de build » pour activer les Options développeur.
  2. Options développeur > activer « Débogage USB ».
  3. Sur le PC : installer adb (paquet `android-platform-tools` / `adb`).
  4. Brancher le téléphone en USB, accepter la fenêtre d'autorisation,
     vérifier avec `adb devices`.
  5. Lancer Mini Metro, démarrer une partie, puis lancer ce script.

Usage :
  python3 run_android.py --capture-only          # sauver des captures (calibration)
  python3 run_android.py --dry-run captures/*.png # tester la vision hors ligne
  python3 run_android.py --agent greedy           # jouer (recommandé d'abord)
  python3 run_android.py --agent dqn              # jouer avec le modèle appris
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import cv2

from android.adb_io import AdbDevice
from android.bridge import GameView, GestureMapper, action_mask, observation
from android.vision import VisionConfig, annotate, parse_screenshot
from agents.greedy_agent import GreedyAgent


class _EnvShim:
    """GreedyAgent attend un objet avec .game : on lui donne la vue écran."""
    def __init__(self, view: GameView) -> None:
        self.game = view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["greedy", "dqn"], default="greedy")
    parser.add_argument("--model", default="models/dqn.pt")
    parser.add_argument("--serial", default=None, help="numéro de série adb si plusieurs appareils")
    parser.add_argument("--period", type=float, default=2.0, help="secondes entre deux décisions")
    parser.add_argument("--save-dir", default="captures", help="dossier des captures annotées")
    parser.add_argument("--capture-only", action="store_true",
                        help="sauvegarde des captures brutes sans jouer (calibration)")
    parser.add_argument("--dry-run", nargs="*", default=None, metavar="IMAGE",
                        help="analyse des images données (pas de téléphone requis)")
    parser.add_argument("--no-act", action="store_true",
                        help="observe et affiche les décisions sans toucher l'écran")
    args = parser.parse_args()

    cfg = VisionConfig()
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------ mode hors-ligne
    if args.dry_run is not None:
        paths = args.dry_run or sorted(glob.glob(os.path.join(args.save_dir, "raw_*.png")))
        if not paths:
            raise SystemExit("Aucune image à analyser. Faites d'abord --capture-only.")
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                print(f"illisible : {path}")
                continue
            state = parse_screenshot(img, cfg)
            out_path = os.path.join(
                args.save_dir, "annot_" + os.path.basename(path)
            )
            cv2.imwrite(out_path, annotate(img, state))
            print(
                f"{path}: {len(state.stations)} stations, "
                f"{len(state.lines)} lignes, "
                f"{sum(s.queue_len for s in state.stations)} passagers "
                f"-> {out_path}"
            )
        return

    # ------------------------------------------------------------ mode téléphone
    device = AdbDevice(args.serial)
    serial = device.check_connection()
    w, h = device.screen_size()
    print(f"Appareil : {serial} ({w}x{h})")

    if args.capture_only:
        print("Capture toutes les 2 s — Ctrl-C pour arrêter.")
        i = 0
        while True:
            img = device.screenshot()
            path = os.path.join(args.save_dir, f"raw_{i:04d}.png")
            cv2.imwrite(path, img)
            print("sauvé :", path)
            i += 1
            time.sleep(2.0)

    if args.agent == "dqn":
        from agents.dqn import DQNAgent
        agent = DQNAgent()
        agent.load(args.model)
    else:
        agent = GreedyAgent()

    mapper = GestureMapper()
    t0 = time.time()
    step = 0
    print("L'IA joue — Ctrl-C pour arrêter.")
    while True:
        loop_start = time.time()
        img = device.screenshot()
        state = parse_screenshot(img, cfg)
        if state.game_over:
            print("Écran de fin détecté, arrêt.")
            cv2.imwrite(os.path.join(args.save_dir, "game_over.png"), img)
            break
        view = GameView(state)
        mask = action_mask(view)

        if args.agent == "dqn":
            obs = observation(view, elapsed_s=time.time() - t0)
            action = agent.act(obs, mask)
        else:
            action = agent.act(None, mask, _EnvShim(view))

        gesture = mapper.to_gesture(view, action) if action != 0 else None
        desc = gesture.label if gesture else ("attente" if action == 0 else "préparation")
        print(
            f"[{step:04d}] stations={len(view.stations)} "
            f"attente={sum(len(s.queue) for s in view.stations)} -> {desc}"
        )
        if gesture and not args.no_act:
            device.drag(gesture.x1, gesture.y1, gesture.x2, gesture.y2)

        if step % 10 == 0:
            cv2.imwrite(
                os.path.join(args.save_dir, f"annot_live_{step:04d}.png"),
                annotate(img, state),
            )
        step += 1
        time.sleep(max(0.0, args.period - (time.time() - loop_start)))


if __name__ == "__main__":
    main()
