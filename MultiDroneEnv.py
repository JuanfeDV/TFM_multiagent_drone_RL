from SingleDroneEnv import SingleDroneEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np
import time
from typing import Set, Dict, Any, Tuple, List
from ultralytics import YOLO
import torch
import cv2
import airsim
import math
import os



class MultiDroneEnv(gym.Env):
    """
    Entorno cooperativo para múltiples drones en AirSim mediante política compartida.
    """

    def __init__(self, drone_ids: List[str], inference_mode: bool = False):
        """
        Inicializa el entorno cooperativo.
        Args:
            drone_ids (List[str]): Lista de ids de los drones.
            inference_mode (bool): Modo inferencia (por defecto False).
        """

        super().__init__()
        self.drone_ids = drone_ids
        self.num_drones = len(drone_ids)
        self.shared_map_cells_explored: Set[Tuple[float, float]] = set()    # Mapa compartido de las celdas exploradas
        self.shared_map_people_detected: Set[Tuple[float, float]] = set()   # Mapa compartido de las celdas donde se ha encontrado una persona


        # Parámetros del mapa y del campo de visión de la cámara
        self.GRID_WIDTH = 25
        self.GRID_DEPTH = 8
        self.MAP_WIDTH = 100
        self.MAP_HEIGHT = 100
        
        # Parámetro global de duración máxima del episodio cooperativo
        self.max_steps = 2000
        self.current_step = 0

        # Parámetros para YOLO
        self.PERSON_DETECTION_THRESHOLD = 0.80
        self.PERSON_DETECTION_RADIUS = 2


        # Se instancia YOLO y se ejecuta en GPU
        self.yolo_model = YOLO("yolov8m.pt")
        if torch.cuda.is_available():
            self.yolo_model.to("cuda:0")


        # Contador de episodios
        self.episode_num = 0



        # Se crea un entorno único por cada dron
        # Los mapas de celdas exploradas y de personas detectadas es común a todos los drones (acceden a misma memoria)
        self.drones: List[SingleDroneEnv] = []
        for name in drone_ids:
            env = SingleDroneEnv(
                drone_name=name,
                inference_mode=inference_mode,
                shared_map_cells_explored=self.shared_map_cells_explored,
                shared_map_people_detected=self.shared_map_people_detected,
                grid_width=self.GRID_WIDTH,
                grid_depth=self.GRID_DEPTH,
            )
            self.drones.append(env)


        # Se inicializan los drones
        for drone in self.drones:
            drone.reset()


        # Se consultan los valores mínimos y máximos del espacio de observaciones de los drones, en este caso normalizado 0-1 y se concatenan los 3 espacios (3 drones)
        low_state = np.tile(self.drones[0].observation_space.low, self.num_drones)
        high_state = np.tile(self.drones[0].observation_space.high, self.num_drones)

        # Se define el espacio de observaciones para el entorno cooperativo con la concatenación de los espacios de observaciones de los 3 drones
        self.observation_space = gym.spaces.Box(
            low=low_state,
            high=high_state,
            dtype=np.float32
        )



        # Se define el espacio de acciones cogiendo el primer dron de plantilla (son todos iguales) y concatenando los 3 espacios
        self.action_space = gym.spaces.MultiDiscrete([
            self.drones[0].action_space.n for _ in range(self.num_drones)
        ])


        # Se almacena la última posición y la ruta trazada de los 3 drones para monitorización (No se usa en el modelo)
        self.last_positions: Dict[str, Tuple[float, float]] = {}
        self.drone_paths: Dict[str, List[Tuple[float, float]]] = {}





    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Resetea todos los drones y devuelve la observación inicial.

        Args:
            seed (int, optional): Semilla para reproducibilidad.
            options (dict, optional): Parámetros adicionales.

        Returns:
            Tuple[np.ndarray, dict]: 
                - Observación inicial (vector de estados concatenados).
                - Diccionario adicional con imágenes ("image": lista de imágenes por dron).
        """

        # Si no se ha recibido semilla, crearla
        if seed is not None:
            np.random.seed(seed)

        #Se inicializan listas para almacenar los estados e imágenes de los 3 drones
        state_vecs, img_vecs = [], []

        # Se borran las listas de última posición y rutas trazadas de los drones
        self.last_positions.clear()
        self.drone_paths.clear()

        # Se resetean los mapas de celdas exploradas y personas encontradas (comunes a todos los drones)
        self.shared_map_cells_explored.clear()
        self.shared_map_people_detected.clear()

        # Se incrementa el número de episodio para llevar un conteo
        self.episode_num += 1

        # Se resetea cada dron y se obtienen sus observaciones iniciales
        for drone in self.drones:
            obs_dict, _ = drone.reset(seed=seed, options=options)
            state_vecs.append(obs_dict['state'])
            frame = obs_dict.get('image') if obs_dict.get('image') is not None else drone.last_frame
            img_vecs.append(frame.reshape(-1)) # Se convierte la imagen a vector



        ## CONTROL DE LOS DRONES

        # Se conecta al cliente de AirSim del primer dron (es el mismo para todos)
        client = self.drones[0].client

        # Se habilitan y arman los drones
        for d in self.drones:
            client.enableApiControl(True, d.drone_name)
            client.armDisarm(True, d.drone_name)

        # Se ordena a los drones despegar y esperar nuevas órdenes
        takeoff_futures = [
            client.takeoffAsync(vehicle_name=d.drone_name)
            for d in self.drones
        ]
        for f in takeoff_futures:
            f.join()

        # Se ordena posición hover a los drones y esperar nuevas órdenes
        hover_futures = [
            client.hoverAsync(vehicle_name=d.drone_name)
            for d in self.drones
        ]
        for f in hover_futures:
            f.join()



        # Se obtienen las posiciones de los 3 drones para su monitorización
        self.spawn_pos = {}
        for d in self.drones:
            st = d.client.getMultirotorState(d.drone_name).kinematics_estimated
            self.spawn_pos[d.drone_name] = (st.position.x_val, st.position.y_val)


        # Se concatenan los vectores de estado de los 3 drones en un solo vector
        vec_state = np.concatenate(state_vecs).astype(np.float32)

        # Se devuelve el vector de estados y el vector de imágenes como info (solo para YOLO y no para el modelo)
        return vec_state, {"image": img_vecs}





    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Ejecuta un paso del entorno cooperativo:
        - Envía acciones a cada dron
        - Comprueba la imagen de cada dron con YOLO a ver si hay personas
        - Calcula métricas y devuelve la observación conjunta

        Args:
            actions (List[int]): Lista de acciones (una por dron)

        Returns:
            Tuple:
                - Observación (np.ndarray): vector concatenado de estados de todos los drones
                - Recompensa global (float)
                - done (bool): indica si el episodio ha terminado
                - truncated (bool): indica si se ha truncado
                - info (dict): información adicional para métricas y visualización
        """

        # Se crean vectores para almacenar los resultados de las acciones realizadas
        state_vecs, img_vecs, rewards, dones = [], [], [], []
        
        # Por cada dron del sistema
        for i, drone in enumerate(self.drones):
            # Se ejecutan la acción de cada dron correspondiente y se recoge el nuevo estado y recompensa
            obs_dict, r, term, trunc, _ = drone.step(actions[i])
            state_vecs.append(obs_dict['state'])

            # Se obtiene la imagen de la cámara del dron. En caso de haber un fallo momentáneo se usa la anterior para continuar
            frame = obs_dict.get('image') if obs_dict.get('image') is not None else drone.last_frame

            # Se aplana la imagen para inferencia en YOLO
            img_vecs.append(frame.reshape(-1))

            # Se almacena la recompensa y si done
            rewards.append(r)
            dones.append(term or trunc)



        # Se calcula la recompensa media obtenida de los drones
        total_reward = float(np.mean(rewards))

        # Se asigna done=True si algún drone ha sido done o se han explorado 2.500 celdas o más
        done = any(dones) or len(self.shared_map_cells_explored) >= 2500

        # Control de steps máximo superado por episodio
        self.current_step += 1
        truncated = self.current_step >= self.max_steps



        ##### YOLO #####

        # Se obtiene la resolución de la imagen de un dron de ejemplo para reconstruirlas y almacenarlas
        H, W, C = self.drones[0].last_frame.shape
        imgs = [v.reshape(H, W, C) for v in img_vecs]

        # Diccionario para almacenar las personas detectadas
        detections = {}

        # Inferencia con YOLO de las imágenes de cada dron
        for i, drone in enumerate(self.drones):
            count_new = self.detect_and_process_people(drone, imgs[i])
            detections[self.drone_ids[i]] = count_new



        ##### MÉTRICAS PARA TENSORBOARD #####

        # Se obtiene la altura promedio de los drones
        heights = [
            d.z_base - d.client.getMultirotorState(d.drone_name).kinematics_estimated.position.z_val
            for d in self.drones
        ]
        avg_height = float(np.mean(heights))

        # Se obtiene el número de colisiones totales de los drones
        num_collisions = sum([
            1 for d in self.drones if d.client.getDistanceSensorData("ProximityFront", d.drone_name).distance < 1.0
        ])

        # Se obtiene el número de personas detectadas por los drones
        num_people_detected = sum(detections.values())

        # Se obtiene el número de celdas nuevas exploradas
        num_cells_explored = len(self.shared_map_cells_explored)


        # Se devuelve info como diccionario con las métricas e imágenes
        # info no es computado por el modelo PPO
        info = {
            "rewards": rewards,
            "avg_height": avg_height,
            "num_collisions": num_collisions,
            "num_people_detected": num_people_detected,
            "num_cells_explored": num_cells_explored,
            "image": img_vecs
        }

        # Se concatena la observación (vector de estados)
        vec_state = np.concatenate(state_vecs).astype(np.float32)
        

        # Retorno estandar tipo gym (obs, rew, done, trunc, info)
        return vec_state, total_reward, done, False, info





    def close(self) -> None:
        """
        Cierra correctamente el entorno cooperativo:
            - Llama al método close() de cada dron para desarmar, liberar control y resetear.
            - Cierra cualquier ventana gráfica abierta de OpenCV.
        """

        for d in self.drones:
            d.close()
        cv2.destroyAllWindows()





    def detect_and_process_people(self, drone: SingleDroneEnv, image_rgb: np.ndarray) -> int:
        """
        Realiza inferencia YOLO en una imagen de un dron y guarda la imagen si detecta una persona nueva.
        La persona se considera nueva si no se ha detectado antes dentro del área de margen.
        
        Args:
            drone (SingleDroneEnv): instancia del dron.
            image_rgb (np.ndarray): imagen RGB capturada por el dron.
        
        Returns:
            int: número de nuevas personas detectadas por este dron en este paso.
        """

        # Se convierte imagen de RGB a BGR para OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Inferencia de la imagen en YOLO
        results = self.yolo_model.predict(image_rgb, verbose=False)[0]
        count_new = 0

        # Por cada caja encontrada en la imagen
        for box in results.boxes:

            # Se recoge la clase de la caja y su nivel de confianza
            cls_name = self.yolo_model.names[int(box.cls.item())]
            conf = box.conf.item()

            # Se comprueba si la caja es person y el nivel de confianza por encima del umbral asignado (80%)
            if cls_name == "person" and conf >= self.PERSON_DETECTION_THRESHOLD:

                # Se obtiene la posición y orientación del dron para estimar la celda del mapa donde se encuentra la persona detectada
                st = drone.client.getMultirotorState(drone.drone_name).kinematics_estimated
                yaw = airsim.to_eularian_angles(st.orientation)[2]
                altura = drone.z_base - st.position.z_val
                pitch_rad = math.radians(abs(drone.CAMERA_PITCH))
                distancia_centro = altura / math.tan(pitch_rad)
                obs_x = st.position.x_val + distancia_centro * math.cos(yaw)
                obs_y = st.position.y_val + distancia_centro * math.sin(yaw)
                cell_x = int(obs_x // drone.grid_width)
                cell_y = int(obs_y // drone.grid_depth)


                # Se verifica si ya existe una persona detectada previamente en un radio definido alrededor de la celda actual
                exists = False
                for dx in range(-self.PERSON_DETECTION_RADIUS, self.PERSON_DETECTION_RADIUS + 1):
                    for dy in range(-self.PERSON_DETECTION_RADIUS, self.PERSON_DETECTION_RADIUS + 1):
                        if (cell_x + dx, cell_y + dy) in self.shared_map_people_detected:
                            exists = True
                            break
                    if exists:
                        break


                # Si la persona no ha sido detectada antes en la zona, se registra la celda, se incrementa el contador y se marca para recompensar al dron
                if not exists:
                    self.shared_map_people_detected.add((cell_x, cell_y))
                    count_new += 1
                    drone.detected_new_person = True


                    ##### ESCRITURA EN IMAGEN #####

                    # Se dibuja el bounding box donde se encuentra la persona
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(image_bgr, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

                    # Se escribe sobre el cuadro el nombre de la clase y el porcentaje de confianza
                    label = f"{cls_name} {conf*100:.1f}%"
                    cv2.putText(
                        image_bgr,
                        label,
                        (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    # Se escribe en la esquina inferior izquierda el nombre del dron y la celda donde se encuentra la persona
                    h, w, _ = image_bgr.shape
                    text = f"{drone.drone_name} Cell=({cell_x},{cell_y})"
                    cv2.putText(
                        image_bgr, 
                        text, 
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0), 
                        2, 
                        cv2.LINE_AA
                    )

                    # Se guarda la imagen con las modificaciones realizadas
                    filename = (
                        f"detections/Episode_{self.episode_num}_"
                        f"Person_{drone.drone_name}_Cell_{cell_x}_{cell_y}_"
                        f"Step_{self.current_step}.png"
                    )
                    cv2.imwrite(filename, image_bgr)


        return count_new
