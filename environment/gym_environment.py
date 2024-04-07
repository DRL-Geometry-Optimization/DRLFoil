import numpy as np
import gymnasium as gym
from gymnasium import spaces

from parametrization import airfoiltools
from reward import reward




class AirfoilEnv(gym.Env):
    metadata = {'render.modes': ["human", "no_display"], "render_fps": 4 }

    def __init__(self, state0, max_iter=300, efficiency_th=None):

        # state0 should have the following structure: [[downparameters],[upparameters],LE_weight]

        super(AirfoilEnv, self).__init__() # Initialize the parent class 

        if len(state0) == 3:
            pass
        else:
            raise ValueError(f"State should be an array of 3 arrays. Length obtained: {len(state0)}")

        #self.upparameters = len(state0[0]) # Number of parameters in the upper side of the airfoil
        #self.downparameters = len(state0[1])
        self.max_iter = max_iter
        self.efficiency_th = efficiency_th 
        self.state = airfoiltools() # Create an airfoil object
        self.state.kulfan(state0[0], state0[1], state0[2]) # Create the airfoil with the Kulfan parameterization
        self.upparameters = self.state.upparameters # Number of parameters in the upper side of the airfoil
        self.downparameters = self.state.downparameters
        self.done = False # The episode is not done by default 
        self.step_counter = 0
        self.reward = 0

        # The action space is the weights of the airfoil. +1 is for the weight of the leading edge
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.upparameters+self.downparameters+1,), dtype=np.float32) 
        
        # The observation space is the airfoil
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=(self.upparameters+self.downparameters+1,), dtype=np.float32)



    def step(self, action):
        """
        This method takes an action and returns the new state, the reward, and whether the episode is done.
        """
        # Update the state of the environment
        self.state.modify_airfoil(action)
        self.state.analysis() # Analyze the airfoil
        self.step_counter += 1
        print(self.state.aerodynamics)
        """
        # Calculate the reward
        self.reward = reward(self.state, self.efficiency_th)

        # Check if the episode is done
        if self.step_counter >= self.max_iter:
            self.done = True

        return self.state, self.reward, self.done, {}
        """






if __name__ == "__main__":
    test = [0.1*np.ones(10),-0.1*np.ones(10),0]
    camion = AirfoilEnv(test)
    camion.step([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],0])
    camion.state.airfoil_plot()
    camion.step([0.1*np.ones(10),-0.1*np.ones(10),0])
    camion.state.airfoil_plot()

    #print(reward(15, 10, delta_reward=False, cl_target=True))