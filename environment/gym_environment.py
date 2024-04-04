import numpy as np
import gymnasium as gym
from gymnasium import spaces

from parametrization import airfoiltools
from reward import reward




class AirfoilEnv(gym.Env):
    metadata = {'render.modes': ["human", "no_display"], "render_fps": 4 }

    def __init__(self, state0, max_iter=300, effic_threshold=None):

        # state0 should have the following structure: [[upparameters],[downparameters],LE_weight]

        super(AirfoilEnv, self).__init__() # Initialize the parent class 

        if len(state0) == 3:
            pass
        else:
            raise ValueError(f"State should be an array of 3 arrays. Length obtained: {len(state0)}")

        self.upparameters = len(state0[0]) # Number of parameters in the upper side of the airfoil
        self.downparameters = len(state0[1])
        self.max_iter = max_iter
        self.effic_threshold = effic_threshold
        self.state = state0
        self.done = False # The episode is not done by default 
        self.step_counter = 0
        self.reward = 0

        # The action space is the weights of the airfoil. +1 is for the weight of the leading edge
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.upparameters+self.downparameters+1,), dtype=np.float32) 
        
        # The observation space is the airfoil
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=(self.upparameters+self.downparameters+1,), dtype=np.float32)


        #self.observation_space = spaces.Dict({
        #    "airfoil": spaces.Box(low=np.NINF, high=np.Inf, shape=(n_params,2), dtype=np.float32), # coordenadas de los puntos del airfoil. Guarda up y down
        #    "aerodynamics": spaces.Box(low=np.NINF, high=np.Inf, shape=(2,), dtype=np.float32), # coeficientes de sustentación y resistencia
        #})

        def step(self, action):
            pass








if __name__ == "__main__":
    test = [[1,2,3,4,5],[1,2,3],2]
    camion = AirfoilEnv(test)
    print(camion.action_space)

    print(reward(15, 10, delta_reward=False, cl_target=True))