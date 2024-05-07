import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from . import airfoil_env

class Optimize: 
    def __init__(self, model : str, cl_target : float, reynolds : float):

        def _find_values(filename, key):
            """
            Function used to find gym environment parameters
            """
            with open(filename, 'r') as file:
                for line in file:
                    if line.startswith(key):
                        return line.split(':')[1].strip()
                    

        allowed_models = ['onebox', 'twobox', 'nobox']  # Available models (using this list to avoid path injection)

        self.model = None # Placeholder for the model. It will be a PPO object
        self.env = None # Placeholder for the environment. It will be a gym environment

        self.cl_target = cl_target
        self.reynolds = reynolds

        if model == 'onebox':
            self.model_path = "models/onebox/onebox.zip"
            self.log_path = "models/onebox/log_onebox.txt"
            self.env = gym.make('AirfoilEnv-v0', 
                           n_params= int(_find_values(self.log_path, 'n_params')),
                           max_steps= int(_find_values(self.log_path, 'max_steps')),
                           scale_actions = float(_find_values(self.log_path, 'scale_actions')),
                           airfoil_seed = [0.1*np.ones(int(_find_values(self.log_path, 'n_params'))), -0.1*np.ones(int(_find_values(self.log_path, 'n_params'))), 0.0], #TO BE CHANGED
                           delta_reward= False, 
                           cl_reward = bool(_find_values(self.log_path, 'cl_reward')),
                           cl_reset = self.cl_target, 
                           efficiency_param = float(_find_values(self.log_path, 'efficiency_param')),
                           cl_wide = float(_find_values(self.log_path, 'cl_wide')),
                           render_mode="human",
                           n_boxes=1,
                           reynolds = self.reynolds)
            self.model = PPO.load(self.model_path, env=self.env)




              

    def load(self,):
        pass

    def optimize(self,):
        pass

    def save(self,):
        pass

    def show(self,):
        pass


if __name__ == '__main__':
    pedro = Optimize('onebox', 0.5, 1000000)
    print(pedro.model)
    pedro.model.reset()