import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AirfoilEnv(gym.Env):
    metadata = {'render.modes': ['human'], "render_fps": 30}

    def __init__(self, n_params=15):
        """
        Se realizan acciones divididas en tres:
        - Escoger si se modifica la superficie superior o inferior
        - Escoger qué parámetro se modifica
        - Escoger si se aumenta o disminuye el parámetro

        En caso de que se pudiesen hacer modificaciones continuas en los parámetros, se podría hacer algo como:
        spaces.Box(low=-0.5, high=0.5, shape=(1,)),  # cantidad a modificar (entre -1 y 1, continuo)
        Si no, se tiene que hacer algo discreto

        Si se juntan todas las acciones en un espacio discreto unico, se puede hacer haciendo combinatoria de acciones.
        Se puede hacer tanto policy-based como value-based. A lo mejor es mas comodo juntar todas las acciones y tener que
        escoger de todas solo una, pero se pierde la estructura de las acciones.
        """
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),  # superficie a modificar: "up" o "down"
            spaces.Discrete(n_params),  # parámetro a modificar
            spaces.Discrete(2)  # dirección de la modificación: "aumentar 0.1" o "disminuir 0.1"
              
        ))


        self.observation_space = spaces.Dict({
            "airfoil": spaces.Box(low=np.NINF, high=np.Inf, shape=(n_params,2), dtype=np.float32), # coordenadas de los puntos del airfoil. Guarda up y down
            "aerodynamics": spaces.Box(low=np.NINF, high=np.Inf, shape=(2,), dtype=np.float32), # coeficientes de sustentación y resistencia
        })



print(AirfoilEnv().observation_space)