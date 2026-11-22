# Historique des versions

Variante Mecanum (holonome) du projet
[limo-offline-learning](https://github.com/jupivato/limo-offline-learning/blob/main/CHANGELOG.md).
Chaque version correspond à une étiquette Git.

## v1.0 : alignement sur le projet différentiel et documentation publique (22/11/2026)

- `config.json` devient la **source unique** de tous les paramètres, lus par un chargeur strict.
- Validation sans fuite (séparation par trajectoire), modèle réentraîné, chargement strict des
  poids, temps de maintien mesuré sur l'horloge murale.
- Télémétrie compacte avec la vitesse latérale $v_y$ ; `run_autonomous.py` réservé au robot
  réel (`--real`).
- Documentation publique : INSTALL, TRAINING, ROBOTS (mesure ISO 18646-2
  manuelle).

## v0.1 : PioneerNN à 3 sorties (18/11/2026)

- Expert holonome, jeu de données de 45 000 exemples, PioneerNN 3 → 1000 → 3 sans biais
  (commande nulle au quai), contrôle de `linear.x`, `linear.y` et `angular.z`.
- README en français, notes matérielles (`use_mcnamu:=true`).
