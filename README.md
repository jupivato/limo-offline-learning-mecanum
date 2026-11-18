# LIMO Offline Learning — Mode Mecanum (Holonome, 3 Sorties)

Projet d'apprentissage hors ligne (*behavioral cloning*) pour la commande holonome et l'amarrage autonome en boucle fermée du robot **AgileX LIMO** en mode **Mecanum**, sous ROS 2 (`/odom` et `/cmd_vel` avec `linear.x`, `linear.y` et `angular.z`).

> **Cible d'amarrage :** $(x = 0.0\text{ m},\; y = 0.0\text{ m},\; \theta = 0.0\text{ rad})$.  
> Le contrôleur lit l'odométrie accumulée des roues et ramène le robot vers l'origine où le pilote a été démarré.

---

## Structure du projet

```text
limo-offline-learning-mecanum/
├── data/
│   └── dataset_mecanum.csv             # Jeu de données holonome (45 000 échantillons, 6 colonnes)
├── models/
│   ├── pioneer_mecanum_weights.json    # Poids entraînés du PioneerNN (3 -> 1000 -> 3)
│   └── training_history_mecanum.json   # Historique de la perte MSE durant l'entraînement
├── src/
│   ├── generate_dataset_mecanum.py     # Générateur du jeu de données (vx, vy, w)
│   ├── train_mecanum.py                # Entraînement hors ligne du réseau de neurones
│   ├── core/
│   │   └── pioneer_nn.py               # Architecture MLP PioneerNN (PyTorch + fallback Python pur)
│   ├── ros2/
│   │   ├── limo_interface.py           # Interface ROS 2 (/odom et /cmd_vel avec linear.y)
│   │   └── show_pose.py                # Moniteur de pose (x, y, theta) en temps réel
│   └── evaluation/
│       ├── data_logger.py              # Enregistrement de la télémétrie à 20 Hz et graphiques
│       └── run_autonomous.py           # Exécution autonome en boucle fermée (défaut --mode mecanum)
└── README.md
```

---

## 1. Générer le jeu de données et entraîner (optionnel)

Le jeu de données (`data/dataset_mecanum.csv`) et les poids entraînés (`models/pioneer_mecanum_weights.json`) sont déjà inclus et prêts à l'emploi. Exécutez les commandes ci-dessous uniquement si vous souhaitez régénérer les données ou réentraîner le modèle :

```bash
python3 src/generate_dataset_mecanum.py
python3 src/train_mecanum.py
```

---

## 2. Exécution sur le robot physique (AgileX LIMO — Mode Mecanum)

> **Important :** Avant de lancer le pilote, assurez-vous que le LIMO physique est bien configuré en **mode Mecanum** et positionné **exactement au point cible $(0, 0, 0)$**, orienté vers l'avant ($0^\circ$).

Dans **tous** les terminaux ouverts sur le robot, chargez d'abord l'environnement ROS 2 correspondant :
- **LIMO COBOT (Ubuntu 20.04 / ROS 2 Foxy) :** `source /opt/ros/foxy/setup.bash`
- **LIMO Standard (Ubuntu 22.04 / ROS 2 Humble) :** `source /opt/ros/humble/setup.bash`

### Terminal 1 — Lancer le pilote des moteurs (`limo_base` en mode Mecanum)
```bash
# LIMO COBOT (Foxy) :
source /opt/ros/foxy/setup.bash
source ~/limo_ros2_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py use_mcnamu:=true

# OU LIMO Standard (Humble) :
source /opt/ros/humble/setup.bash
source ~/agilex_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py use_mcnamu:=true
```
*(Laissez ce terminal ouvert pendant toute la durée du test. Si le port série `ttylimo` n'est pas détecté, ajoutez `port_name:=ttyUSB0`. Sur le LIMO COBOT, exécutez les commandes directement sur le robot — en local ou via SSH — car `ROS_LOCALHOST_ONLY=1` évite les plantages de CycloneDDS.)*

### Terminal 2 — Téléopération (amener le robot au point de départ)
```bash
source /opt/ros/foxy/setup.bash   # ou humble
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
* `i` / `,` : Avant / Arrière
* `j` / `l` : Rotation sur place (gauche / droite)
* `k` : Arrêt immédiat

Conduisez le robot jusqu'à la position de départ souhaitée (par ex. à $1{,}5\text{ m}$ ou $2{,}0\text{ m}$ de distance), appuyez sur **`k`** pour l'arrêter, puis quittez avec **`Ctrl + C`**.

### Terminal 3 — Lancer le contrôleur autonome Mecanum
```bash
source /opt/ros/foxy/setup.bash   # ou humble
cd ~/limo-offline-learning-mecanum
python3 src/evaluation/run_autonomous.py --real --speed-scale 0.6
```
1. Vérifiez la pose initiale affichée depuis `/odom`.
2. Appuyez sur **`Entrée`** pour démarrer la navigation autonome vers $(0, 0, 0)$.
3. Les journaux et graphiques d'exécution sont enregistrés automatiquement dans `data/evaluation/`.

> **Astuce :** Pour réduire davantage la vitesse et limiter le patinage des rouleaux Mecanum sur sol lisse, utilisez `--speed-scale 0.4` ou `0.5`.

### Terminal 4 (optionnel) — Surveiller la pose en temps réel
```bash
source /opt/ros/foxy/setup.bash   # ou humble
cd ~/limo-offline-learning-mecanum
python3 src/ros2/show_pose.py
```

---

## Commandes utiles et arrêt d'urgence

| Action | Commande |
| :--- | :--- |
| **Arrêt d'urgence immédiat** | `ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0}, angular: {z: 0.0}}" --once` |
| **Vérifier la fréquence d'odométrie** | `ros2 topic hz /odom` (ou `/wheel/odom`) |
| **Tester le modèle différentiel avec ce script** | `python3 src/evaluation/run_autonomous.py --real --mode diff --weights <chemin_poids_diff.json>` |
