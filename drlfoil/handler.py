import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from drlfoil import airfoil_env
import copy

class Optimize: 
    def __init__(self, model : str, cl_target : float, reynolds : float, logs : int = 1):
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

        if logs not in [0, 1, 2, 3]:
            raise ValueError("Logs level must be 0, 1, 2 or 3")
        self.logs = logs # Define the level of logs to be shown
        
        self.model = None # Placeholder for the model. It will be a PPO object
        self.env = None # Placeholder for the environment. It will be a gym environment

        self.cl_target = cl_target
        self.reynolds = reynolds

        self.bestairfoil = None # Placeholder for the best airfoil found

        if model == 'onebox':
            self.model_path = "models/onebox/onebox.zip"
            self.log_path = "models/onebox/log_onebox.txt"

            if self.logs >= 1:
                print("*** Loading model from", self.model_path, "***")

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

            if self.logs >= 2:
                print("*** Model loaded in", time.time()-start_time, "seconds")

        elif model == 'twobox':
            self.model_path = "models/twobox/twobox.zip"
            self.log_path = "models/twobox/log_twobox.txt"

            if self.logs >= 1:
                print("*** Loading model from", self.model_path, "***")
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
                           n_boxes=2,
                           reynolds = self.reynolds)
            self.model = PPO.load(self.model_path, env=self.env)
            if self.logs >= 2:
                print("*** Model loaded in", time.time()-start_time, "seconds")

        elif model == 'nobox':
            self.model_path = "models/nobox/nobox.zip"
            self.log_path = "models/nobox/log_nobox.txt"

            if self.logs >= 1:
                print("*** Loading model from", self.model_path, "***")

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
                           n_boxes=0,
                           reynolds = self.reynolds)
            self.model = PPO.load(self.model_path, env=self.env)

            if self.logs >= 2:
                print("*** Model loaded in", time.time()-start_time, "seconds")


    def run(self, ):
        """
        Function used to optimize the airfoil environment using the pre-trained model.
        Logic: The function runs the model for a number of episodes defined by the env. 
        The best airfoil found is saved in the bestairfoil attribute.
        """
        start_time = time.time()
        done = False
        obs, _ = self.env.reset()
        self.bestairfoil = {'airfoil': None, 'reward': -np.inf}
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.env.step(action)

            if self.logs == 3:
                print("*** Airfoil found with an efficiency of", info['efficiency'], "and lift coefficient of", info['cl'])

            if reward > self.bestairfoil['reward']:
                self.bestairfoil = {'airfoil': copy.deepcopy(self.env.unwrapped.state), 
                                    'reward': reward, 
                                    'efficiency': info['efficiency'], 
                                    'cl': info['cl']}

                if self.logs == 3:
                    print("*** New best airfoil found!")

        if self.logs >= 1:
            print("*** Optimization finished! Time elapsed:", time.time()-start_time, "seconds")
            print(f"     Best airfoil found with a reward of {self.bestairfoil['reward']}, lift coefficient of {self.bestairfoil['cl']} and efficiency of {self.bestairfoil['efficiency']}")   
            self.bestairfoil['airfoil'].airfoil_plot()
            print(self.bestairfoil['airfoil'].aerodynamics)

    def save(self,):

        airfoil_coords = self.bestairfoil['airfoil'].get_coordinates()
        with open("BORRAAAAR.dat", 'w') as f:
            f.write("Airfoil coordinates\n")

            for item in range(len(airfoil_coords[0])):
                # Pass the first element since it is the same as the last element of the upper surface
                f.write(str(airfoil_coords[0][item][0])+ '   '+str(airfoil_coords[0][item][1])+ '\n')

            for item in range(len(airfoil_coords[1])):
                # Pass the first element since it is the same as the last element of the upper surface
                if item == 0:
                    continue
                f.write(str(airfoil_coords[1][item][0])+ '   '+str(airfoil_coords[1][item][1])+ '\n')



            


    def show(self,):
        pass


if __name__ == '__main__':
    pedro = Optimize('onebox', 0.5, 1000000)
    print(pedro.model)
    pedro.model.reset()