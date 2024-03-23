import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AirfoilEnv(gym.Env):
    metadata = {'render.modes': ['human'], "render_fps": 30}

    def __init__(self, n_params=15):
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),  # superficie a modificar: "up" o "down"
            spaces.Discrete(n_params),  # parámetro a modificar
            spaces.Discrete(2)  # dirección de la modificación: "aumentar 0.1" o "disminuir 0.1"
            
            """
            En caso de que se pudiesen hacer modificaciones continuas en los parámetros, se podría hacer algo como:
            spaces.Box(low=-0.5, high=0.5, shape=(1,)),  # cantidad a modificar (entre -1 y 1, continuo)
            Si no, se tiene que hacer algo discreto
            """
        ))



print(AirfoilEnv().action_space)