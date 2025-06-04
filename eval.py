from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from MultiDroneEnv import MultiDroneEnv
from utils import get_latest_checkpoint
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any



# Parámetros de la evaluación
PHASE = "phase3"
MODEL_ROOT = Path(f"models/{PHASE}")
MODEL_PATH = MODEL_ROOT / "model_final.zip"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
DRONES = ["Drone1", "Drone2", "Drone3"]
N_EPISODES = 20
MAX_STEPS = 2_000




def load_model(env) -> PPO:
    """ Carga el modelo entrenado desde modelo final o último checkpoint """
    if MODEL_PATH.exists():
        return PPO.load(MODEL_PATH, env=env)
    ckpt = get_latest_checkpoint(str(CHECKPOINT_DIR))
    if ckpt:
        return PPO.load(ckpt, env=env)
    raise FileNotFoundError("No se encontró modelo entrenado")



def unwrap(env):
    """ Elimina los wrappers para acceder al entorno base (MultiDroneEnv) """
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env
    return raw



def run_episode(model: PPO, env) -> Dict[str, Any]:
    """ Ejecuta un episodio de evaluación completo """

    # Inicialización del entorno y variables de control
    obs, _ = env.reset()
    done, trunc = False, False
    ep_reward = 0.0
    steps = 0

    # Bucle de interacción hasta terminar el episodio
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, _ = env.step(action)
        ep_reward += reward
        steps += 1

    # Cálculo de métricas finales al acabar el episodio
    raw_env = unwrap(env)
    people_total = len(raw_env.shared_map_people_detected)
    cells_total = len(raw_env.shared_map_cells_explored)

    # Se calcula el total de personas detectadas, celdas exploradas y colisiones
    collisions = sum(
        1
        for d in raw_env.drones
        if d.client.getDistanceSensorData("ProximityFront", d.drone_name).distance < 1.0
    )

    return {
        "reward": ep_reward,
        "steps": steps,
        "cells_explored": cells_total,
        "people_detected": people_total,
        "collisions": collisions,
    }



def main() -> None:

    # Se instancia el entorno de evaluación
    env = TimeLimit(
        MultiDroneEnv(DRONES, inference_mode=True),
        max_episode_steps=MAX_STEPS,
    )

    # Se carga el modelo entrenado
    model = load_model(env)

    # Bucle principal de evaluación por episodios
    results: List[Dict[str, Any]] = []
    for ep in range(1, N_EPISODES + 1):
        print(f"\nEpisodio {ep}/{N_EPISODES}")
        metrics = run_episode(model, env)

        # Almacena métricas del episodio y muestra resumen por pantalla
        results.append(metrics)
        print(
            f" Recompensa: {metrics['reward']:.1f} | "
            f"Pasos: {metrics['steps']} | "
            f"Celdas: {metrics['cells_explored']} | "
            f"Personas: {metrics['people_detected']} | "
            f"Colisiones: {metrics['collisions']}"
        )


    # Se guardan los resultados en CSV
    df_out = pd.DataFrame(results)
    df_out.index += 1  # para que empiece en 1
    df_out.index.name = "episode"
    csv_name = f"eval_results_{PHASE}.csv"
    df_out.to_csv(csv_name)
    print(f"\nResultados guardados en {csv_name}")

    # Se calcula y muestra el promedio de cada métrica
    avg = {k: np.mean(df_out[k]) for k in df_out.columns}
    print("\nPromedios:")
    for k, v in avg.items():
        print(f" {k:15}: {v:.2f}" if isinstance(v, (float, int)) else f" {k:15}: {v}")



if __name__ == "__main__":
    main()
