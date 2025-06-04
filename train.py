from MultiDroneEnv import MultiDroneEnv
from callbacks import Metrics_callback, get_checkpoint_callback, get_eval_callback
from utils import get_latest_checkpoint
import os, time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from codecarbon import EmissionsTracker



if __name__ == "__main__":

    # Parámetros del entrenamiento
    PHASE         = "phase3"
    TIMESTEPS     = 5_200_000
    MODEL_ROOT    = f"models/{PHASE}"
    MODEL_PATH    = f"{MODEL_ROOT}/model_final"
    CHECKPOINT_DIR= f"{MODEL_ROOT}/checkpoints"
    TBOARD_ROOT   = f"./tensorboard/{PHASE}"


    # Se crean los directorios necesarios si no existen
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TBOARD_ROOT, exist_ok=True)


    # Se instancia el entorno de entrenamiento
    drone_ids = ["Drone1", "Drone2", "Drone3"]
    train_env = Monitor(MultiDroneEnv(drone_ids, inference_mode=False))



    ##### MODELO #####

    # Se busca último checkpoint disponible
    latest = get_latest_checkpoint(CHECKPOINT_DIR)

    # Si existe modelo final, se carga
    if os.path.exists(f"{MODEL_PATH}.zip"):
        print("##### Reanudando modelo guardado #####")
        model = PPO.load(MODEL_PATH, env=train_env)

    # Si no existe modelo final, se escoge el último checkpoint
    elif latest:
        print(f"##### Reanudando desde {latest} #####")
        model = PPO.load(latest, env=train_env)

    # Si no existe modelo final ni checkpoints, se instancia el modelo desde cero
    else:
        print("##### Comenzando entrenamiento #####")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            tensorboard_log=TBOARD_ROOT,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            learning_rate=1e-3,
            ent_coef=1e-3,
            device="cuda",
            policy_kwargs=dict(net_arch=[128, 128])
        )



    # Instancia de entorno de evaluación
    raw_eval = MultiDroneEnv(drone_ids, inference_mode=True)
    raw_eval = TimeLimit(raw_eval, max_episode_steps=2000)
    eval_env = Monitor(raw_eval, filename=None, allow_early_resets=True)


    # Callbacks
    checkpoint_cb = get_checkpoint_callback(CHECKPOINT_DIR)
    eval_cb = get_eval_callback(eval_env, MODEL_ROOT)


    # Se mide el consumo energético durante el entrenamiento con CodeCarbon
    tracker_CC = EmissionsTracker()
    tracker_CC.start()



    ##### ENTRENAMIENTO #####

    # En caso de haber cargado checkpoint, se carga el número de steps completados
    num_steps_done = getattr(model, "num_timesteps", 0)
    num_steps_left = max(TIMESTEPS - num_steps_done, 0)
    print(f"##### Modelo con {num_steps_done} steps completados, faltan {num_steps_left} #####")


    # Se Inicia o continua el entrenamiento
    start = time.time()
    model.learn(
        total_timesteps=num_steps_left,
        callback=[checkpoint_cb, eval_cb, Metrics_callback()],
        progress_bar=True,
        log_interval=1_000,
        reset_num_timesteps=False,
        tb_log_name=PHASE
    )
    print(f"##### Entrenamiento completado en {time.time()-start:.1f}s #####")


    # Se guarda el modelo final entrenado
    final_path = f"{MODEL_ROOT}/model_final_{model.num_timesteps}_steps"
    model.save(final_path)
    print(f"##### Modelo guardado en: {final_path}.zip #####")


    # Se guarda el log de CodeCarbon
    tracker_CC.stop()