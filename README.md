# LIMO Offline Learning — Mecanum (3 Outputs: vx, vy, w)

Projeto separado para controle holonômico do robô **AgileX LIMO** em modo **Mecanum** via ROS 2 (`/odom` e `/cmd_vel` com `linear.x`, `linear.y` e `angular.z`).

## Estrutura

- `src/generate_dataset_mecanum.py` — Gera `data/dataset_mecanum.csv` (45.000 amostras, 6 colunas).
- `src/train_mecanum.py` — Treina a `PioneerNN` (`3 -> 1000 -> 3`) e salva em `models/pioneer_mecanum_weights.json`.
- `src/ros2/limo_interface.py` — Interface ROS 2 com suporte a `linear.x`, `linear.y` e `angular.z`.
- `src/evaluation/run_autonomous.py` — Execução autônoma em malha fechada (padrão `--mode mecanum`).
- `src/ros2/show_pose.py` — Monitor de pose em tempo real.

## Uso no robô físico (Mecanum)

1. Terminal 1 — Driver do LIMO (com o robô posicionado no alvo `(0, 0, 0)`):
```bash
ros2 launch limo_base limo_base.launch.py
```

2. Terminal 2 — Teleoperação até o ponto de partida desejado:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

3. Terminal 3 — Executar navegação autônoma Mecanum:
```bash
cd ~/limo-offline-learning-mecanum
python3 src/evaluation/run_autonomous.py --real --speed-scale 0.6
```
