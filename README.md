# LIMO Offline Learning (Mecanum) : apprentissage hors-ligne pour le LIMO en mode holonome

Ce dépôt est la variante **Mecanum** (holonome) du projet
[limo-offline-learning](https://github.com/jupivato/limo-offline-learning) : un superviseur
expert génère un jeu de données, un réseau **PioneerNN** (MLP 3 → 1000 → 3, tanh) apprend
à l'imiter, puis le réseau seul ramène le robot AgileX LIMO, en mode Mecanum, jusqu'au quai
d'amarrage $(0, 0, 0^\circ)$ en commandant `linear.x`, `linear.y` et `angular.z`.

| Dépôt | Mode | Sorties du réseau | Simulation Gazebo | Évaluation |
|---|---|---|---|---|
| [`limo-offline-learning`](https://github.com/jupivato/limo-offline-learning) | Différentiel (non-holonome) | `v, ω` (2) | Oui | Gazebo + robot réel |
| **`limo-offline-learning-mecanum`** (ce dépôt) | Mecanum (holonome) | `vx, vy, ω` (3) | Non, pas de modèle officiel | Robot réel |

Il n'existe pas de modèle Gazebo officiel du LIMO en mode Mecanum : ce dépôt se teste
**uniquement sur le robot réel**.

## Comment ça marche

```
config.json ─► generate_dataset_mecanum.py ─► data/dataset_mecanum.csv ─► train_mecanum.py ─► models/pioneer_mecanum_weights.json
                (expert holonome)              (45 000 exemples)          (PioneerNN 3->1000->3)        │
                                                                                                        ▼
                                                              run_autonomous.py --real (robot en mode Mecanum)
```

- Entrées (repère du robot) : $[x_r/5,\ y_r/5,\ e_\theta/\pi]$.
- Sorties : $[\hat v_x, \hat v_y, \hat\omega] \in [-1, 1]^3$, converties en
  $v_x = \hat v_x \cdot 0.50 \cdot s$, $v_y = \hat v_y \cdot 0.50 \cdot s$ (norme limitée à $0.50 \cdot s$),
  $\omega = \hat\omega \cdot 1.00$, avec $s$ = `speed_scale` = 0,55.
- Réseau sans biais : la commande est exactement nulle au quai, $\text{MLP}(0, 0, 0) = (0, 0, 0)$.
- Tous les paramètres sont dans **`config.json`**, seule source de vérité.

## Pour commencer

1. [`docs/INSTALL.md`](docs/INSTALL.md) : installer l'environnement (ROS 2, dépendances Python)
2. [`docs/TRAINING.md`](docs/TRAINING.md) : générer le jeu de données, entraîner le réseau, tracer les courbes
3. [`docs/ROBOTS.md`](docs/ROBOTS.md) : faire tourner le contrôleur sur le robot et mesurer ses performances (ISO 18646-2)

Le jeu de données et les poids entraînés sont déjà inclus : pour une démonstration, on peut
passer directement à l'essai sur le robot.

## Démarrage rapide sur le LIMO COBOT (ROS 2 Foxy)

Tout se lance **sur le robot**. Dans chaque nouveau terminal, répondre `2` à la question
`ros: noetic(1) foxy(2) ?`. Le robot est posé sur le quai (cap à 0°), roues Mecanum montées.

```bash
# Terminal 1 : driver en mode Mecanum (le quai = position du robot à cet instant)
source /opt/ros/foxy/setup.bash && source ~/limo_ros2_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py use_mcnamu:=true

# Terminal 2 : téléopération pour éloigner le robot (k = arrêt, puis Ctrl + C)
source /opt/ros/foxy/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Terminal 3 : contrôleur (Entrée pour démarrer)
source /opt/ros/foxy/setup.bash
cd ~/limo-offline-learning-mecanum
python3 src/evaluation/run_autonomous.py --real
```

Sur le LIMO standard (Humble), remplacer `foxy` par `humble` et `~/limo_ros2_ws` par
`~/agilex_ws`. Détails, dépannage et mesure ISO 18646-2 : [`docs/ROBOTS.md`](docs/ROBOTS.md).

## Structure du dépôt

```
config.json                       tous les paramètres du projet (source unique)
requirements.txt                  dépendances Python
CHANGELOG.md                      historique des versions
src/
  generate_dataset_mecanum.py     superviseur expert holonome + génération du jeu de données
  train_mecanum.py                entraînement supervisé de la PioneerNN à 3 sorties
  core/config.py                  chargement strict de config.json, critères d'amarrage
  core/pioneer_nn.py              architecture PioneerNN
  ros2/limo_interface.py          interface ROS 2 (/odom, /cmd_vel avec linear.y)
  ros2/show_pose.py               affichage de la pose en direct dans le terminal
  evaluation/run_autonomous.py    essai d'amarrage en boucle fermée (--mode mecanum par défaut)
  evaluation/data_logger.py       enregistrement CSV 20 Hz et graphiques
data/dataset_mecanum.csv          jeu de données (45 000 exemples, 6 classes)
models/                           poids entraînés et historique d'apprentissage
docs/                             documentation
```

## Contexte

Projet Pluridisciplinaire en Équipe (PPE),
Centre Européen de Réalité Virtuelle (CERV) / École Nationale d'Ingénieurs de Brest (ENIB).
