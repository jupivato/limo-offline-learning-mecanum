# LIMO Offline Learning — Modo Mecanum (Holonômico, 3 Saídas)

Projeto de aprendizado offline (*behavioral cloning*) para controle holonômico e amarragem autônoma em malha fechada do robô **AgileX LIMO** em modo **Mecanum**, utilizando ROS 2 (`/odom` e `/cmd_vel` com `linear.x`, `linear.y` e `angular.z`).

> **Alvo de Amarrage:** $(x = 0.0\text{ m},\; y = 0.0\text{ m},\; \theta = 0.0\text{ rad})$.  
> O controlador lê a odometria acumulada das rodas e leva o robô de volta à origem onde o driver foi iniciado.

---

## Estrutura do Projeto

```text
limo-offline-learning-mecanum/
├── data/
│   └── dataset_mecanum.csv             # Dataset holonômico (45.000 amostras, 6 colunas)
├── models/
│   ├── pioneer_mecanum_weights.json    # Pesos treinados da PioneerNN (3 -> 1000 -> 3)
│   └── training_history_mecanum.json   # Histórico de perda MSE do treinamento
├── src/
│   ├── generate_dataset_mecanum.py     # Gerador do dataset (vx, vy, w)
│   ├── train_mecanum.py                # Treinamento offline da rede neural
│   ├── core/
│   │   └── pioneer_nn.py               # Arquitetura MLP PioneerNN (PyTorch + fallback Python puro)
│   ├── ros2/
│   │   ├── limo_interface.py           # Interface ROS 2 (/odom e /cmd_vel com linear.y)
│   │   └── show_pose.py                # Monitor de pose (x, y, theta) em tempo real
│   └── evaluation/
│       ├── data_logger.py              # Registro de telemetria a 20 Hz e gráficos
│       └── run_autonomous.py           # Execução autônoma em malha fechada (padrão --mode mecanum)
└── README.md
```

---

## 1. Gerar Dataset e Treinar (Opcional)

O dataset (`data/dataset_mecanum.csv`) e os pesos treinados (`models/pioneer_mecanum_weights.json`) já estão incluídos prontos para uso. Só execute os comandos abaixo se quiser regenerar ou retreinar:

```bash
python3 src/generate_dataset_mecanum.py
python3 src/train_mecanum.py
```

---

## 2. Execução no Robô Físico (AgileX LIMO — Modo Mecanum)

> **Importante:** Antes de ligar o driver, certifique-se de que o LIMO físico está configurado no **modo Mecanum** e posicionado **exatamente no ponto alvo $(0, 0, 0)$**, alinhado para frente ($0^\circ$).

Em **todos** os terminais abertos no robô, carregue primeiro o ambiente ROS 2 correspondente:
- **LIMO COBOT (Ubuntu 20.04 / ROS 2 Foxy):** `source /opt/ros/foxy/setup.bash`
- **LIMO Standard (Ubuntu 22.04 / ROS 2 Humble):** `source /opt/ros/humble/setup.bash`

### Terminal 1 — Iniciar o Driver dos Motores (`limo_base`)
```bash
# LIMO COBOT (Foxy):
source /opt/ros/foxy/setup.bash
source ~/limo_ros2_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py

# OU LIMO Standard (Humble):
source /opt/ros/humble/setup.bash
source ~/agilex_ws/install/setup.bash
ros2 launch limo_base limo_base.launch.py
```
*(Deixe este terminal aberto durante todo o teste).*

### Terminal 2 — Teleoperação (Levar o Robô ao Ponto de Partida)
```bash
source /opt/ros/foxy/setup.bash   # ou humble
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
* `i` / `,` : Frente / Ré
* `j` / `l` : Girar no lugar (esquerda / direita)
* `k` : Parar imediatamente

Conduza o robô até a posição inicial desejada (ex.: $1.5\text{ m}$ ou $2.0\text{ m}$ de distância), aperte **`k`** para parar e saia com **`Ctrl + C`**.

### Terminal 3 — Executar o Controlador Autônomo Mecanum
```bash
source /opt/ros/foxy/setup.bash   # ou humble
cd ~/limo-offline-learning-mecanum
python3 src/evaluation/run_autonomous.py --real --speed-scale 0.6
```
1. Confira a pose inicial exibida a partir do `/odom`.
2. Pressione **`Enter`** para iniciar a navegação autônoma até $(0, 0, 0)$.
3. Os logs e gráficos da execução serão salvos automaticamente em `data/evaluation/`.

> **Dica:** Para reduzir ainda mais a velocidade e evitar patinagem dos roletes mecanum no piso liso, use `--speed-scale 0.4` ou `0.5`.

### Terminal 4 (Opcional) — Monitorar a Pose em Tempo Real
```bash
source /opt/ros/foxy/setup.bash   # ou humble
cd ~/limo-offline-learning-mecanum
python3 src/ros2/show_pose.py
```

---

## Comandos Úteis e Emergência

| Ação | Comando |
| :--- | :--- |
| **Parada de emergência imediata** | `ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0}, angular: {z: 0.0}}" --once` |
| **Verificar frequência da odometria** | `ros2 topic hz /odom` (ou `/wheel/odom`) |
| **Testar o modelo diferencial neste script** | `python3 src/evaluation/run_autonomous.py --real --mode diff --weights <caminho_pesos_diff.json>` |
