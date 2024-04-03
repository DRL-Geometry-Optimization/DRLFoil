import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AirfoilEnv(gym.Env):
    metadata = {'render.modes': ["human", "no_display"], "render_fps": 4 }

    def __init__(self, state0, max_iter=300, effic_threshold=None):
        super(AirfoilEnv, self).__init__() # Initialize the parent class 
        self.upparameters = (len(state)//2) -1 # Number of parameters in the upper side of the airfoil
        self.max_iter = max_iter
        self.effic_threshold = effic_threshold
        self.state = state0
        self.done = False # The episode is not done by default 
        self.step_counter = 0
        self.reward = 0

        # The action space is the weights of the airfoil. +1 is for the weight of the leading edge
        self.action_space = spaces.Box(low=-1, high=1, shape=(upparameters+downparameters+1,), dtype=np.float32) 
        
        # The observation space is the airfoil
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=(upparameters+downparameters+1,), dtype=np.float32)


        #self.observation_space = spaces.Dict({
        #    "airfoil": spaces.Box(low=np.NINF, high=np.Inf, shape=(n_params,2), dtype=np.float32), # coordenadas de los puntos del airfoil. Guarda up y down
        #    "aerodynamics": spaces.Box(low=np.NINF, high=np.Inf, shape=(2,), dtype=np.float32), # coeficientes de sustentación y resistencia
        #})



print(AirfoilEnv().observation_space)