# Robot physique (AgileX LIMO en mode Mecanum)

Le contrôleur lit l'odométrie (`/odom`, ou `/wheel/odom`) et publie sur `/cmd_vel`
`linear.x`, `linear.y` et `angular.z`. Le quai d'amarrage $(0, 0, 0^\circ)$ est **l'endroit
où se trouve le robot au démarrage du driver**.

## Les deux robots du CERV

| | **LIMO COBOT** (avec bras) | **LIMO** (standard) |
|---|---|---|
| Système | Ubuntu 20.04, Python 3.8 | Ubuntu 22.04, Python 3.10 |
| ROS 2 | Foxy | Humble |
| Workspace du driver | `~/limo_ros2_ws` | `~/agilex_ws` |
| À l'ouverture d'un terminal | le robot demande `ros: noetic(1) foxy(2) ?` : taper **`2`** | rien à faire |
| Où lancer les commandes | **sur le robot** (écran et clavier, ou SSH) : sa découverte ROS 2 est limitée à la machine locale (`ROS_LOCALHOST_ONLY=1`) | sur le robot ou depuis un PC du même réseau |

Sur le LIMO COBOT, `~/agilex_ws` existe aussi mais c'est un workspace **ROS 1** : ne pas l'utiliser.

## 1. Préparer le robot (une seule fois)

Copier le dépôt dans le dossier personnel du robot, avec `git clone` si le robot a internet,
sinon avec une clé USB :

```bash
git clone https://github.com/jupivato/limo-offline-learning-mecanum.git ~/limo-offline-learning-mecanum
```

PyTorch est déjà installé sur les deux robots. Vérifier simplement les dépendances, **sans
lancer `pip install -r requirements.txt`**, qui pourrait remplacer les versions du système :

```bash
python3 -c "import torch, matplotlib; print('torch', torch.__version__, '| OK')"
```

Si seul `matplotlib` manque, l'essai fonctionne quand même (le graphique final n'est pas tracé).

## 2. Lancer un amarrage (3 terminaux sur le robot)

Monter les roues Mecanum et poser le robot **exactement sur le quai** (cap aligné à 0°) :
l'odométrie est mise à zéro au démarrage du driver.

### Terminal 1 : driver en mode Mecanum

**LIMO COBOT** (répondre `2` à la question `ros: noetic(1) foxy(2) ?`) :

```bash
source /opt/ros/foxy/setup.bash
source ~/limo_ros2_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py use_mcnamu:=true
```

**LIMO** :

```bash
source /opt/ros/humble/setup.bash
source ~/agilex_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py use_mcnamu:=true
```

`use_mcnamu` est bien l'orthographe du paramètre dans le driver AgileX. Le port série
(`ttylimo`) est détecté automatiquement. Laisser ce terminal ouvert pendant toute la séance.

### Terminal 2 : téléopération (pour éloigner le robot du quai)

```bash
source /opt/ros/foxy/setup.bash        # LIMO : /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Touches : `i` / `,` avant / arrière, `j` / `l` rotation sur place, `k` arrêt. Une fois le
robot sur sa pose de départ, appuyer sur `k` puis quitter avec `Ctrl + C`.

### Terminal 3 : contrôleur PioneerNN

```bash
source /opt/ros/foxy/setup.bash        # LIMO : /opt/ros/humble/setup.bash
cd ~/limo-offline-learning-mecanum
python3 src/evaluation/run_autonomous.py --real
```

Le script affiche la pose de départ lue sur l'odométrie et attend `Entrée`. Le réseau ramène
le robot au quai puis vérifie 30 s d'immobilité (8 cm, 15°). Télémétrie, graphique et
`params.json` sont enregistrés dans `data/evaluation/test_AAAA_MM_DD_HH_MM_SS/`.

Vitesse : `speed_scale` = 0,55 dans `config.json` (norme de la vitesse ≤ 0,275 m/s). Pour
limiter le patinage des rouleaux sur sol lisse : `--speed-scale 0.4`.

### En cas de problème

| Symptôme | Cause probable | Solution |
|---|---|---|
| `Package 'limo_base' not found` | workspace non sourcé, ou mauvais workspace | COBOT : `source ~/limo_ros2_ws/install/setup.bash` (pas `~/agilex_ws`) |
| `/opt/ros/humble/setup.bash: No such file` | commande Humble sur le COBOT | utiliser `/opt/ros/foxy/setup.bash` |
| `Waiting for valid odometry stream...` en boucle | driver arrêté, ou script lancé depuis un PC (COBOT) | relancer le terminal 1 ; sur le COBOT, tout lancer sur le robot |
| le robot avance mais ne se déplace pas latéralement | driver lancé sans `use_mcnamu:=true` ou roues standard | relancer le driver avec `use_mcnamu:=true` |

## 3. Récupérer les résultats sans réseau commun

Les métriques ISO 18646-2 **n'ont besoin d'aucun fichier du robot** : les poses atteintes
sont mesurées au sol et notées à la main dans la feuille de mesures, directement sur le PC.
Les fichiers du robot (`data/evaluation/test_*`) ne servent qu'aux trajectoires et à la
vérification.

Sans SSH (PC et robot pas sur le même réseau), le plus simple est une **clé USB** branchée
sur le robot (clavier et écran, ou session locale) :

```bash
lsblk                                   # repérer la clé (ex. /media/agilex/CLE)
cp -r ~/limo-offline-learning-mecanum/data/evaluation/test_* /media/agilex/CLE/
sync                                    # attendre la fin de l'écriture avant de retirer la clé
```

Alternatives : câble Ethernet direct entre le PC et le robot (puis `scp`), ou connexion du
PC au point d'accès Wi-Fi du robot s'il en a un. Sur le PC, copier les dossiers dans
`data/evaluation/` du dépôt, puis générer les figures avec `report_figures.py --sessions`.

## 4. Outils utiles

| Action | Commande |
|---|---|
| Pose en direct | `python3 src/ros2/show_pose.py` |
| Arrêt d'urgence | `ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0}, angular: {z: 0.0}}" --once` |
| Fréquence de l'odométrie | `ros2 topic hz /odom` |
| Tester un modèle différentiel sur ce robot | `python3 src/evaluation/run_autonomous.py --real --mode diff --weights chemin/vers/poids_diff.json` |
| Partir d'une pose marquée au sol | `python3 src/evaluation/run_autonomous.py --real --teleport-to X Y THETA` (THETA en radians, recale l'odométrie) |

Si le robot publie à la fois `/odom` et `/wheel/odom`, fixer `ros2_and_safety.odom_topic`
dans `config.json` pour que tous les outils lisent la même odométrie.

## 5. Mesurer la précision (ISO 18646-2)

La norme ISO 18646-2 exige que la pose atteinte soit mesurée par un **système externe** :
l'odométrie ne mesure que ce que le robot *croit*. Sans système de capture de mouvement, on
mesure au sol :

1. Marquer au sol l'origine (0, 0), un axe X (direction 0°) et un axe Y.
2. Coller deux repères sur le robot : centre de l'avant (F) et centre de l'arrière (T).
3. Marquer au sol la position de F et T pour chaque pose de départ, et repartir **exactement
   de la même pose** à chaque répétition (robot replacé à la main, puis `--teleport-to X Y THETA`).
4. À l'arrivée, mesurer F et T : position = milieu de F et T, cap $\theta = \text{atan2}(y_F - y_T,\ x_F - x_T)$.
5. Noter une ligne par essai (10 répétitions par pose) dans une copie de la feuille
   [`iso_manual_measurements_template.csv`](https://github.com/jupivato/limo-offline-learning/blob/main/docs/iso_manual_measurements_template.csv)
   du dépôt différentiel.
6. Calculer $A_p$, $A_o$, $R_p$, $R_o$ par pose avec la calculatrice du dépôt différentiel
   (le calcul est identique en mode Mecanum) :
   ```bash
   cd ~/limo-offline-learning
   python3 calculus/calculate_iso_metrics.py --manual ~/mes_mesures_mecanum.csv
   ```
