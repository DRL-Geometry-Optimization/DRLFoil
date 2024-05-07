import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from drlfoil import airfoil_env

class Optimize: 
    def __init__(self, model : str, cl_target : float, reynolds : float):
        """
        Class used to handle the optimization of the airfoil environment with the pre-trained models

        Args:
        -model: Define the model to be used. Available models are: 'onebox', 'twobox', 'nobox' 
        -cl_target: Define the target lift coefficient
        -reynolds: Define the Reynolds number of the flow
        """

        def _find_values(filename : str, key: str):
            """
            Function used to find gym environment parameters. It is defined to follow the recorder module file format!

            Args:
            -filename: Path to the log file
            -key: Key to be found in the log file
            """
            with open(filename, 'r') as file:
                for line in file:
                    if line.startswith(key):
                        return line.split(':')[1].strip()
                    

        allowed_models = ['onebox', 'twobox', 'nobox']  # Available models (using this list to avoid path injection)
        if model not in allowed_models:
            raise ValueError(f"Model {model} not found. Available models are: {allowed_models}")

        self.model = None # Placeholder for the model. It will be a PPO object
        self.env = None # Placeholder for the environment. It will be a gym environment

        self.cl_target = cl_target
        self.reynolds = reynolds

        if model == 'onebox':
            self.model_path = "models/onebox/onebox.zip"
            self.log_path = "models/onebox/log_onebox.txt"
            print("********** Loading model from", self.model_path, "**********")
            start_time = time.time()
            # Load the environment with the parameters from the log file and cl targey & reynolds defined by the user
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
            print("********** Model loaded in", time.time()-start_time, "seconds **********")




              

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