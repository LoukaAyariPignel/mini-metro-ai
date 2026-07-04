# Mini Metro AI 🚇

Une IA qui **apprend à jouer à Mini Metro** par apprentissage par renforcement
(Deep Q-Network), entraînée sur une simulation fidèle du jeu écrite de zéro.

> Mini Metro étant un jeu propriétaire sans API, ce projet recrée les règles du
> jeu sous forme de simulation, l'expose comme environnement RL, puis entraîne
> un agent à y jouer. L'IA s'améliore par essai-erreur — elle ne joue pas
> « parfaitement » (personne ne connaît le jeu parfait de Mini Metro, la
> difficulté monte à l'infini), mais elle apprend seule des stratégies qui
> écrasent le hasard et rivalisent avec une heuristique écrite à la main.

## La simulation (`mini_metro/game.py`)

Règles reproduites du vrai jeu :

- Des **stations** apparaissent progressivement sur la carte, chacune avec une
  forme : cercle ○, triangle △, carré □, plus les formes rares étoile ★ et
  pentagone ⬠ après la première semaine.
- Des **passagers** apparaissent dans les stations, chacun voulant rejoindre
  une station de la forme qu'il porte. Le rythme d'apparition augmente avec le
  temps (la difficulté monte).
- Le joueur trace jusqu'à **3 lignes de métro** ; chaque ligne fait circuler
  des **trains** (capacité 6) en aller-retour, qui chargent et déposent les
  passagers.
- Si une station **déborde** (plus de 6 passagers en attente) pendant 30
  secondes : **partie perdue**.
- Chaque **semaine** de jeu, un train supplémentaire est accordé.

Le score est le nombre de passagers livrés.

## L'environnement RL (`mini_metro/env.py`)

Interface compatible Gymnasium :

- **Observation** (vecteur de 264 flottants) : position, forme, file d'attente
  et surcharge de chaque station ; composition des lignes ; charge des trains ;
  temps et pression globale.
- **Actions** (52 discrètes) : ne rien faire, effacer une ligne, ou étendre la
  ligne *l* vers la station *s*. Un **masque d'actions** interdit les coups
  illégaux.
- **Récompense** : +1 par passager livré, bonus quand une station est reliée
  pour la première fois, malus par station en surcharge, −20 en cas de défaite.

## Les agents (`agents/`)

| Agent | Description |
|---|---|
| `RandomAgent` | joue au hasard — la baseline plancher |
| `GreedyAgent` | heuristique à la main : relie les stations orphelines, étend les lignes vers les formes demandées |
| `DQNAgent` | **Double DQN** avec masque d'actions, replay buffer et réseau cible — apprend seul |

## Utilisation

```bash
pip install -r requirements.txt

# Entraîner l'IA (≈ 250 épisodes, sauvegarde dans models/dqn.pt)
python3 train.py --episodes 250

# Comparer les agents sur les mêmes parties
python3 evaluate.py --episodes 20

# Regarder une partie en ASCII dans le terminal
python3 play.py --agent greedy
python3 play.py --agent dqn
```

## Résultats

Sur 20 parties identiques (mêmes graines), mesuré avec `evaluate.py` :

| Agent | Score moyen | Survie moyenne | Meilleur score |
|---|---|---|---|
| Aléatoire | 42.6 ± 35.8 | 442 s (meurt vite) | 123 |
| DQN (250 épisodes, ~5 min CPU) | 137.4 ± 56.0 | 710 s | 277 |
| Glouton | 670.4 ± 158.0 | 1724 s | 763 |

Après seulement 250 épisodes, l'IA a appris seule à tripler le score de
l'agent aléatoire et à survivre presque deux fois plus longtemps. La courbe
d'apprentissage est dans `logs/train_log.csv` (score et survie par épisode).
Pour la pousser plus loin : entraîner plus longtemps (`--episodes 2000`),
agrandir le réseau dans `agents/dqn.py`, ou passer à PPO.

## Structure

```
mini_metro/
  game.py    # moteur du jeu (stations, lignes, trains, passagers)
  env.py     # environnement RL (observation, actions, récompense)
agents/
  random_agent.py
  greedy_agent.py
  dqn.py     # Double DQN + replay buffer
train.py     # boucle d'entraînement
evaluate.py  # comparaison des agents
play.py      # visualisation ASCII d'une partie
```
