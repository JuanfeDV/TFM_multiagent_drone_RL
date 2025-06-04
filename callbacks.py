from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter



class Metrics_callback(BaseCallback):
    """
    Callback para registrar métricas en TensorBoard.

    Args:
        log_freq (int): Frecuencia en pasos.
        verbose (int): Nivel de detalle de logs.
    """
    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.writer: SummaryWriter = None
        self.step_count = 0


    def _on_training_start(self) -> None:
        # Inicializa el writer
        self.writer = SummaryWriter(log_dir=self.logger.dir)


    def _on_step(self) -> bool:
        # Se ejecuta en cada step del entrenamiento y registra las métricas según frecuencia
        self.step_count += 1
        if self.step_count % self.log_freq != 0:
            return True

        # Accede al entorno
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env

        # Se obtiene la info de la observación
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # Se obtienen las métricas solicitadas
        info = infos[0] 
        avg_height = info.get("avg_height")
        num_collisions = info.get("num_collisions")
        num_people_detected = info.get("num_people_detected")
        num_cells_explored = info.get("num_cells_explored")

        # Se comprueba si los valores no están vacios y se registra en Tensorboard
        if avg_height is not None:
            self.writer.add_scalar("Team/Mean_Height_m", float(avg_height), int(self.num_timesteps))
        if num_collisions is not None:
            self.writer.add_scalar("Team/Collision_Count", int(num_collisions), int(self.num_timesteps))
        if num_people_detected is not None:
            self.writer.add_scalar("Team/New_People_Detected_Per_Step", int(num_people_detected), int(self.num_timesteps))
        if num_cells_explored is not None:
            self.writer.add_scalar("Team/Explored_Cells_Count", int(num_cells_explored), int(self.num_timesteps))
        if hasattr(env, "shared_map_people_detected"):
            total_detected = len(env.shared_map_people_detected)
            self.writer.add_scalar("Team/Total_People_Detected", int(total_detected), int(self.num_timesteps))


        # Se continua el entrenamiento con el siguiente step
        return True



    def _on_training_end(self) -> None:
        # Se cierra el writer al finalizar el entrenamiento
        self.writer.close()




def get_checkpoint_callback(checkpoint_dir: str) -> CheckpointCallback:
    """
    Devuelve un callback para guardar checkpoints periódicos.

    Args:
        checkpoint_dir (str): Ruta donde se guardarán los checkpoints.

    Returns:
        CheckpointCallback: Callback configurado.
    """
    return CheckpointCallback(
        save_freq=5_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_multi_coop"
    )



def get_eval_callback(eval_env, model_root: str) -> EvalCallback:
    """
    Devuelve un callback para evaluar periódicamente el modelo.

    Args:
        eval_env: Entorno ya instanciado para evaluación.
        model_root (str): Carpeta raíz para guardar mejores modelos y logs.

    Returns:
        EvalCallback: Callback de evaluación.
    """
    return EvalCallback(
        eval_env,
        best_model_save_path=f"{model_root}/best_model/",
        log_path=f"{model_root}/eval_logs/",
        eval_freq=100_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
