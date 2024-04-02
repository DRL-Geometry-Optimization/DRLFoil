import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AirfoilEnv(gym.Env):
    metadata = {'render.modes': ["human", "no_display"], "render_fps": 4 }

    def __init__(self, n_params=15):
        super(AirfoilEnv, self).__init__() # Initialize the parent class 
        self.n_params = n_params
        


        self.observation_space = spaces.Dict({
            "airfoil": spaces.Box(low=np.NINF, high=np.Inf, shape=(n_params,2), dtype=np.float32), # coordenadas de los puntos del airfoil. Guarda up y down
            "aerodynamics": spaces.Box(low=np.NINF, high=np.Inf, shape=(2,), dtype=np.float32), # coeficientes de sustentación y resistencia
        })



print(AirfoilEnv().observation_space)