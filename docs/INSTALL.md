# Installation

L'entraînement n'a besoin que de Python. ROS 2 n'est nécessaire que pour piloter le robot ;
sur les LIMO du CERV, ROS 2 et le driver sont déjà installés (voir [`ROBOTS.md`](ROBOTS.md)).

## 1. Python

```bash
sudo apt install python3 python3-pip
git clone https://github.com/jupivato/limo-offline-learning-mecanum.git
cd limo-offline-learning-mecanum
pip install -r requirements.txt      # torch, matplotlib
```

## 2. ROS 2 (machine de développement, optionnel)

Pour utiliser `show_pose.py` ou la téléopération depuis un ordinateur relié au robot,
installer ROS 2 Humble (Ubuntu 22.04) ou Foxy (Ubuntu 20.04). Les commandes complètes sont
dans le [`INSTALL.md` du dépôt différentiel](https://github.com/jupivato/limo-offline-learning/blob/main/docs/INSTALL.md) ;
en résumé :

```bash
sudo apt install ros-humble-desktop ros-humble-teleop-twist-keyboard
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

Gazebo n'est pas utile : il n'existe pas de modèle officiel du LIMO en mode Mecanum.

## 3. Sur le robot (LIMO COBOT ou LIMO)

ROS 2, le driver et PyTorch sont déjà installés sur les robots du CERV : il suffit de copier
le dépôt et de vérifier les dépendances, **sans `pip install -r requirements.txt`** (voir
[`ROBOTS.md`](ROBOTS.md#1-préparer-le-robot-une-seule-fois)). Le mode Mecanum se choisit
au lancement du driver (`use_mcnamu:=true`).

## 4. Vérifier l'installation

```bash
python3 src/train_mecanum.py --epochs 1 --weights-out /tmp/w.json --history-out /tmp/h.json
```

L'entraînement d'une époque doit se terminer sans erreur (sans toucher au modèle publié).
