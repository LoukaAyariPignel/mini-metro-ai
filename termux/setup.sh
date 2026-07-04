#!/data/data/com.termux/files/usr/bin/bash
# Installation sur le téléphone (Termux) — à lancer une seule fois.
#
#   bash termux/setup.sh
#
# Prérequis : Termux installé depuis F-Droid (PAS le Play Store, version morte).
set -e

echo "== Mise à jour des paquets Termux =="
pkg update -y

echo "== Installation : python, adb, numpy, opencv =="
pkg install -y python android-tools git
# opencv et numpy précompilés par Termux (pip les compilerait pendant des heures)
pkg install -y python-numpy opencv-python || pkg install -y python-numpy python-opencv

echo "== Vérification =="
python3 -c "import numpy, cv2; print('numpy', numpy.__version__, '| opencv', cv2.__version__)"
adb version | head -1

echo
echo "Installation terminée."
echo "Étape suivante : bash termux/play.sh (voir le README, section Termux)."
