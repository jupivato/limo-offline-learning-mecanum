# Jeu de données et entraînement

Toutes les commandes se lancent depuis la racine du dépôt. Aucune n'a besoin de ROS 2.
Le jeu de données et les poids sont déjà inclus ; ces étapes ne servent qu'à les regénérer.

## 0. Les paramètres : `config.json`

`config.json` est la **seule source** des paramètres ; une clé manquante arrête le programme
avec un message explicite. Blocs : `kinematics` (vitesses, `speed_scale` = 0,55),
`docking_and_iso` (tolérances 8 cm / 15°, maintien 30 s), `expert_and_dataset` (seuils de
l'expert, grille, 7 500 exemples par classe), `network_and_training` (architecture 3 → 1000 → 3
sans biais, 30 époques, AdamW, batch 128, taux par groupe, graine), `ros2_and_safety` (topics).

## 1. Générer le jeu de données

```bash
python3 src/generate_dataset_mecanum.py
```

L'expert holonome commande, pour chaque état $(x_r, y_r, e_\theta)$ :
- une **translation directe vers la cible** : $(v_x, v_y) = v \cdot (x_r, y_r)/d$, avec une rampe de
  freinage sur `d_brake` = 0,45 m, nulle sous `d_stop_internal` = 1,2 cm ;
- une **rotation** vers le cap cible, avec une rampe sur `angle_brake` = 0,60 rad.

Les états sont équilibrés entre 6 classes (arrêt, rotation pure gauche/droite, translation
pure, translation + rotation gauche/droite), 7 500 exemples par classe.

Sortie : `data/dataset_mecanum.csv` (45 000 lignes), colonnes
`x_target_robot, y_target_robot, theta_robot, vx_robot, vy_robot, w_robot, group_id`.

Les classes de rotation pure ont peu d'états distincts : elles sont complétées par des
**copies bruitées** (±0,5 mm) d'états existants (environ 16 900 copies, nombre affiché par le
terminal). Chaque copie garde le `group_id` de son état d'origine, pour que la séparation
entraînement/validation ne mélange jamais une copie et son original.

## 2. Entraîner le réseau

```bash
python3 src/train_mecanum.py
```

Options : `--epochs N`, `--dataset chemin.csv`, et pour ne pas écraser le modèle publié :
`--weights-out /tmp/w.json --history-out /tmp/h.json`.

Durée : environ 20 s sur CPU. Le terminal affiche à chaque époque la perte d'entraînement et
de validation et l'erreur sur $v_x$, $v_y$ et $\omega$. Le meilleur modèle (perte de validation
minimale) est enregistré.

| Fichier | Contenu |
|---|---|
| `models/pioneer_mecanum_weights.json` | poids du réseau |
| `models/training_history_mecanum.json` | historique par époque et `effective_config` (valeurs réellement utilisées) |

## 3. Courbes d'apprentissage

L'historique est compatible avec le traceur du dépôt différentiel
[limo-offline-learning](https://github.com/jupivato/limo-offline-learning) :

```bash
python3 ../limo-offline-learning/src/visualization/plot_curves.py \
    --history models/training_history_mecanum.json --outdir data/evaluation
```

Produit `data/evaluation/training_curves.png` et `.svg` (perte d'entraînement et de
validation, taux d'apprentissage).
