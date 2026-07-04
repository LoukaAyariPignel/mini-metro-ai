#!/data/data/com.termux/files/usr/bin/bash
# Connecte adb au téléphone lui-même (débogage sans fil) puis lance l'IA.
#
#   bash termux/play.sh                # agent glouton (recommandé)
#   bash termux/play.sh dqn            # agent DQN (inférence numpy)
#   bash termux/play.sh greedy --no-act  # observer sans toucher l'écran
#
# Le débogage sans fil doit être ACTIVÉ (Options développeur > Débogage sans
# fil) et l'appairage déjà fait une première fois (voir README).
set -e

AGENT="${1:-greedy}"
shift 2>/dev/null || true

# garde le processus en vie quand l'écran affiche le jeu
command -v termux-wake-lock >/dev/null && termux-wake-lock || true

if ! adb devices | awk 'NR>1 && $2=="device"' | grep -q .; then
    echo "Aucune connexion adb active."
    echo "Ouvrez Options développeur > Débogage sans fil (gardez-le en écran"
    echo "partagé avec Termux) et lisez 'Adresse IP et port', ex. 192.168.1.10:37099"
    read -rp "Port affiché (juste le nombre après ':') : " PORT
    adb connect "localhost:${PORT}"
fi

adb devices
echo
echo "Lancez Mini Metro, démarrez une partie, puis revenez ici (ou laissez"
echo "ce script tourner : il attend 8 s avant de commencer)."
sleep 8

exec python3 run_android.py --agent "${AGENT}" "$@"
