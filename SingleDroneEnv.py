import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import cv2
import os
import math
from typing import Optional, Set
from typing import Dict, Any



class SingleDroneEnv(gym.Env):
    """
    Entorno individual para un único dron en AirSim.
    """

    # Parámetros del entorno
    DT = 1.0             # Duración en segundos de todas las acciones
    SPEED_FWD = 2.0      # 2 m/s de avance frontal
    SPEED_Z = 1.0        # 1 m/s de velocidad de ascenso
    YAW_RATE = 45.0      # 45°/s de giro
    CAMERA_PITCH = -60.0 # 60° de ángulo de la cámara frontal hacia abajo
    FOV_VERTICAL = 90.0  # 90° campo de visión


    def __init__(
        self,
        drone_name: str,
        shared_map_cells_explored: Set,
        shared_map_people_detected: Set,
        inference_mode: bool = False,
        grid_width: int = 25,
        grid_depth: int = 8
    ):
        super().__init__()
        self.shared_map_cells_explored = shared_map_cells_explored
        self.shared_map_people_detected = shared_map_people_detected
        self.grid_width = grid_width
        self.grid_depth = grid_depth
        self.inference_mode = inference_mode
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.drone_name = drone_name
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        self.MAP_WIDTH  = 100.0
        self.MAP_HEIGHT = 100.0
        self.detected_new_person = False


        # Se define el espacio de observaciones para el entorno individual normalizado [0,1]
        self.observation_space = spaces.Box(
            low = np.zeros(4, dtype=np.float32),
            high= np.ones( 4, dtype=np.float32),
            dtype=np.float32
        )

        # Se define el espacio de acciones
        self.action_space = spaces.Discrete(6)


        # Control del episodio
        self.max_steps = 2_000
        self.current_step = 0
        self.episode_reward = 0.0
        self.z_base = 0.0


        # Se almacenan las detecciones
        self.save_detections = True
        os.makedirs("detections", exist_ok=True)





    def reset(self, seed=None, options=None):
        """
        Resetea el entorno del dron para iniciar un nuevo episodio.
        Reinicia la posición, z_base, pasos, recompensa y mapas compartidos.

        Args:
            seed (int, optional): Semilla para reproducibilidad.
            options (dict, optional): Opcional. Parámetros adicionales.

        Returns:
            Tuple[dict, dict]:
                - dict: Diccionario de observaciones (incluye 'state' y 'image').
                - dict: Diccionario vacío (para compatibilidad con Gymnasium).
        """

        # Reset de gym
        super().reset(seed=seed)

        # Reset del estado del dron en AirSim
        self.client.reset()
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)

        # Se obtiene la altura para referenciar al suelo
        state = self.client.getMultirotorState(self.drone_name).kinematics_estimated
        self.z_base = state.position.z_val

        # Reseteo de contadore
        self.current_step = 0
        self.episode_reward = 0.0

        # Se obtiene la primera observación
        obs = self.get_observation()


        return obs, {}





    def step(self, action: int) -> tuple[Dict[str, Any], float, bool, bool, dict]:
        """
        Ejecuta un paso en el entorno individual del dron:
        - Aplica la acción recibida (movimiento, giro o hover)
        - Calcula la nueva observación
        - Calcula la recompensa basada en exploración, altura, detecciones y colisiones
        - Indica si el episodio ha finalizado o ha sido truncado

        Args:
            action (int): Acción a ejecutar (valor discreto de 0 a 5).

        Returns:
            Tuple:
                - dict: Observación (state + image)
                - float: Recompensa obtenida en este paso
                - bool: terminated → True si se cumple una condición de finalización
                - bool: truncated → True si se alcanza el límite máximo de steps
                - dict: Información adicional (no usado)
        """

        # Se cancelan las posibles acciones en curso y ejecuta la acción correspondiente
        self.client.cancelLastTask(vehicle_name=self.drone_name)

        if action == 0:
            self.client.moveByVelocityBodyFrameAsync(0, 0, -self.SPEED_Z, self.DT, vehicle_name=self.drone_name).join()
        elif action == 1:
            self.client.moveByVelocityBodyFrameAsync(0, 0, self.SPEED_Z, self.DT, vehicle_name=self.drone_name).join()
        elif action == 2:
            self.client.moveByVelocityBodyFrameAsync(self.SPEED_FWD, 0, 0, self.DT, vehicle_name=self.drone_name).join()
        elif action == 3:
            self.client.rotateByYawRateAsync(+self.YAW_RATE, self.DT, vehicle_name=self.drone_name).join()
        elif action == 4:
            self.client.rotateByYawRateAsync(-self.YAW_RATE, self.DT, vehicle_name=self.drone_name).join()
        else:
            self.client.hoverAsync(vehicle_name=self.drone_name).join()

        # Se obtiene la observación, valores (state) normalizados e imagen
        obs_dict = self.get_observation()
        state_vec = obs_dict["state"]

        # Se incrementa el número de step
        self.current_step += 1

        # Reseteo del flag terminated
        terminated = False

        # Se obtienen los valores de state y desnormalizan
        height_n, dist_front_n, x_n, y_n = state_vec
        height = height_n * 10.0
        dist_front = dist_front_n * 40.0


        ##### REWARDS #####

        # Coste temporal
        reward = -0.5

        # Nueva zona vista si está en rango de altura
        if 5.0 <= height <= 8.0:
            st = self.client.getMultirotorState(self.drone_name).kinematics_estimated
            _, _, yaw = airsim.to_eularian_angles(st.orientation)
            newly_cells_visited_count = self.mark_camera_vision(st.position, yaw)
            reward += 0.5 * newly_cells_visited_count


        # Altura. Bonificación por estar en rango y penalización cuadrática fuera de él
        LOW, HIGH = 5.0, 8.0
        MIN_ALT, MAX_ALT = 0.0, 10.0
        k_quad = 3.0

        diff_low = max(LOW - height, 0.0)
        diff_high = max(height - HIGH, 0.0)
        reward -= k_quad * (diff_low**2 + diff_high**2)

        if LOW <= height <= HIGH:
            # centro +- 1 m
            if HIGH-2.0 <= height <= LOW+2.0:
                reward += 2.0
            else:
                reward += 1.0

        if height <= MIN_ALT or height >= MAX_ALT:
            reward -= 200.0
            terminated = True


        # Se bonifica el avance para fomentar exploración
        if action == 2 and LOW <= height <= HIGH:
            reward += 0.5

        # Se penaliza la acción hover para fomentar movimiento
        if action == 5:
            reward -= 1.0

        # Se penaliza obstáculos en trayectoria de movimiento a menos de 2 metros
        if dist_front < 2.0:
            reward -= 50.0

        # Se penaliza obstáculos en trayectoria de movimiento a menos de 0,5 metros y finaliza el episodio
        if dist_front < 0.5:
            reward -= 200.0
            terminated = True


        # Se bonifica la detección de una persona no detectada anteriormente
        # self.detected_new_person obtiene valor True si la imagen del dron ha detectado una persona desde el entorno cooperativo (donde se ejecuta YOLO)
        # Una vez recompensado, se asigna valor False de nuevo hasta que YOLO vuelva a detectar otra persona
        if self.detected_new_person:
            reward += 100.0
            self.detected_new_person = False  

        # Truncated del modo inferencia
        if self.inference_mode:
            truncated = False
        else:
            truncated = self.current_step >= self.max_steps

        # Incremento del número de episodios
        self.episode_reward += reward


        return obs_dict, reward, terminated, truncated, {}





    def get_observation(self) -> dict:
        """
        Obtiene la observación actual del dron:
        - Lee los sensores y la posición del dron en el entorno AirSim
        - Captura la imagen de la cámara frontal inclinada
        - Normaliza todas las variables para formar el vector de estado

        Returns:
            dict:
                - 'state': np.ndarray con los valores normalizados [height, proximity, x, y]
                - 'image': frame RGB capturado por la cámara (o None si no disponible)
        """

        # Se obtiene el estado del dron
        state = self.client.getMultirotorState(self.drone_name).kinematics_estimated
        
        # Se obtiene la altura en referencia al suelo
        height = self.z_base - state.position.z_val

        # Se obtiene la distancia al obstáculo de enfrente (max 40 m)
        proximity = self.client.getDistanceSensorData("ProximityFront", self.drone_name)
        dist_front = proximity.distance if proximity.distance > 0 else 40.0

        # Se obtiene la imagen de la cámara frontal inclinada, se convierte a array de numpy y a BGR
        img = self.client.simGetImages([
            airsim.ImageRequest("front_down", airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.drone_name)[0]

        frame = None

        if img.width > 0:
            arr = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
            if arr.size == img.height * img.width * 3:
                frame = arr.reshape(img.height, img.width, 3).copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_frame = frame


        # Se obtiene la posición del dron en el mapa
        x, y = state.position.x_val, state.position.y_val

        # Se normaliza el tamaño del mapa a [0,1]
        xn = np.clip(x / self.MAP_WIDTH,  0.0, 1.0)
        yn = np.clip(y / self.MAP_HEIGHT, 0.0, 1.0)

        # Se normaliza la altura y el sensor frontal a [0,1]
        height_n     = np.clip(height     / 10.0, 0.0, 1.0)
        dist_front_n = np.clip(dist_front / 40.0, 0.0, 1.0)

        # Vector final de estado con valores normalizados [0,1] de altura, distancia frontal y posición del dron
        state_vec = np.array([
            height_n,
            dist_front_n,
            xn,
            yn
        ], dtype=np.float32)


        # Retorno en formato diccionario del vector de estado y la imagen
        # La imagen no es tratada por el modelo, únicamente por YOLO
        return {
            "state": state_vec,
            "image": frame
        }





    def close(self):
        """
        Cierra correctamente el entorno individual del dron:
            - Desarma el dron y libera el control API en AirSim.
            - Reinicia la simulación del dron.
            - Cierra cualquier ventana gráfica de OpenCV si existiera.
        """
        self.client.armDisarm(False, self.drone_name)
        self.client.enableApiControl(False, self.drone_name)
        self.client.reset()
        cv2.destroyAllWindows()





    def mark_camera_vision(self, position, yaw: float) -> int:
        """
        Marca en el mapa compartido todas las celdas del área de visión estimada de la cámara del dron.

        Se calcula:
        - La proyección del haz de la cámara sobre el suelo.
        - El área de cobertura aproximada en forma de círculo (en base al campo de visión vertical y altura).

        Args:
            position: Objeto position del dron en AirSim (posición 3D).
            yaw (float): Ángulo de orientación horizontal del dron en radianes.

        Returns:
            int: Número de nuevas celdas marcadas como visitadas en el mapa compartido para cálculo de recompensa.
        """

        # Se calcula la altura relativa y distancia al centro del haz
        height = self.z_base - position.z_val
        pitch_rad = math.radians(abs(self.CAMERA_PITCH))
        dist_center = height / math.tan(pitch_rad)

        # Se calcula el punto central del área de visión
        obs_x = position.x_val + dist_center * math.cos(yaw)
        obs_y = position.y_val + dist_center * math.sin(yaw)

        # Tamaño de celda y cálculo de radio aproximado de cobertura de visión
        CELL_SIZE = 2.0
        total_width = 2 * dist_center
        vision_radius = math.ceil((total_width / 2) / CELL_SIZE)

        # Cálculo de celda central
        cell_x = int(obs_x // CELL_SIZE)
        cell_y = int(obs_y // CELL_SIZE)

        # Marcado de celdas vistas alrededor de la celda central según el radio de cobertura
        newly_cells_visited_count = 0
        for dx in range(-vision_radius, vision_radius + 1):
            for dy in range(-vision_radius, vision_radius + 1):
                cell = (cell_x + dx, cell_y + dy)
                if cell not in self.shared_map_cells_explored:
                    self.shared_map_cells_explored.add(cell)
                    newly_cells_visited_count += 1


        return newly_cells_visited_count
