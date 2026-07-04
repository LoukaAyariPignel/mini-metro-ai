"""Entrées/sorties ADB : capture d'écran et injection de gestes tactiles.

Fonctionne sur un téléphone NON rooté : il suffit d'activer le débogage USB
(Options pour les développeurs) et de brancher le téléphone en USB.

Les drags utilisent `input motionevent` (Android 12+, donc OK sur Pixel 7a) ;
repli automatique sur `input swipe` si indisponible.
"""

from __future__ import annotations

import subprocess
import time

import cv2
import numpy as np


class AdbDevice:
    def __init__(self, serial: str | None = None) -> None:
        self.serial = serial
        self._has_motionevent: bool | None = None

    def _adb(self, *args: str, binary: bool = False, timeout: float = 15.0):
        cmd = ["adb"]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += list(args)
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"adb {' '.join(args)} a échoué : {result.stderr.decode(errors='replace')}"
            )
        return result.stdout if binary else result.stdout.decode(errors="replace")

    # ------------------------------------------------------------------ infos

    def check_connection(self) -> str:
        out = self._adb("devices")
        lines = [l for l in out.strip().splitlines()[1:] if l.strip()]
        devices = [l.split()[0] for l in lines if l.split()[-1] == "device"]
        if not devices:
            raise RuntimeError(
                "Aucun appareil ADB connecté. Vérifiez : câble USB, débogage USB "
                "activé, et acceptez la fenêtre d'autorisation sur le téléphone.\n"
                f"Sortie de `adb devices` :\n{out}"
            )
        if self.serial is None:
            self.serial = devices[0]
        return self.serial

    def screen_size(self) -> tuple[int, int]:
        out = self._adb("shell", "wm", "size")
        # "Physical size: 1080x2400"
        for token in out.split():
            if "x" in token:
                w, h = token.split("x")
                return int(w), int(h)
        raise RuntimeError(f"Taille d'écran illisible : {out!r}")

    # ---------------------------------------------------------------- capture

    def screenshot(self) -> np.ndarray:
        """Capture l'écran, retourne une image BGR (OpenCV)."""
        png = self._adb("exec-out", "screencap", "-p", binary=True)
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Capture d'écran illisible (PNG corrompu ?)")
        return img

    # ----------------------------------------------------------------- gestes

    def tap(self, x: int, y: int) -> None:
        self._adb("shell", "input", "tap", str(int(x)), str(int(y)))

    def _supports_motionevent(self) -> bool:
        if self._has_motionevent is None:
            out = self._adb("shell", "input")
            self._has_motionevent = "motionevent" in out
        return self._has_motionevent

    def drag(
        self,
        x1: int, y1: int, x2: int, y2: int,
        duration_ms: int = 600,
        steps: int = 12,
    ) -> None:
        """Drag précis : appui, déplacement progressif, relâchement.

        C'est le geste qui trace/étend une ligne dans Mini Metro.
        """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if self._supports_motionevent():
            self._adb("shell", "input", "motionevent", "DOWN", str(x1), str(y1))
            pause = duration_ms / 1000.0 / max(steps, 1)
            for k in range(1, steps + 1):
                t = k / steps
                mx = int(x1 + (x2 - x1) * t)
                my = int(y1 + (y2 - y1) * t)
                self._adb("shell", "input", "motionevent", "MOVE", str(mx), str(my))
                time.sleep(pause)
            self._adb("shell", "input", "motionevent", "UP", str(x2), str(y2))
        else:
            self._adb(
                "shell", "input", "swipe",
                str(x1), str(y1), str(x2), str(y2), str(duration_ms),
            )
