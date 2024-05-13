import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from drlfoil import airfoil_env
from drlfoil.utilities import AeroAnalysis
import copy

import os



class Optimize: 
    def __init__(self, model : str, cl_target : float, reynolds : float, boxes : list = [], steps : int = 10, logs : int = 1):
        """
        Class used to handle the optimization of the airfoil environment with the pre-trained models

        Args:
        -model: Define the model to be used. Available models are: 'onebox', 'twobox', 'nobox' 
        -cl_target: Define the target lift coefficient
        -reynolds: Define the Reynolds number of the flow
        -boxes: Define the box restrictions for the airfoil. It is a list of BoxRestriction objects
        -steps: Define the number of steps to run the optimization
        -logs: Define the level of logs to be shown. 0: No logs, 1: Simple optimization data, 2: Simple optimization data and optimization time, 3: All about the optimization and the airfoils found
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

        self.steps = steps

        self.bestairfoil = None # Placeholder for the best airfoil found
        self.bestairfoil_reward = -np.inf

        self.boxes = boxes

        if model == 'onebox':

            # Obtiene la ruta del directorio donde se encuentra el script actual
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Construye la ruta al modelo de manera relativa al script
            self.model_path = os.path.join(current_dir, "models", "onebox", "onebox.zip")
            #self.model_path = "drlfoil/models/onebox/onebox.zip"
            #self.log_path = "drlfoil/models/onebox/log_onebox.txt"
            self.log_path = os.path.join(current_dir, "models", "onebox", "log_onebox.txt")

            if self.logs >= 1:
                print("*** Loading model from", self.model_path, "***")

            if len(self.boxes) != 1:
                raise ValueError("Onebox model requires one box restriction")

            start_time = time.time()
            # Load the environment with the parameters from the log file and cl targey & reynolds defined by the user
            self.env = gym.make('AirfoilEnv-v0', 
                           n_params= int(_find_values(self.log_path, 'n_params')),
                           max_steps= steps,
                           scale_actions = float(_find_values(self.log_path, 'scale_actions')),
                           airfoil_seed = [0.1*np.ones(int(_find_values(self.log_path, 'n_params'))), -0.1*np.ones(int(_find_values(self.log_path, 'n_params'))), 0.0], #TO BE CHANGED
                           delta_reward= False, 
                           cl_reward = bool(_find_values(self.log_path, 'cl_reward')),
                           cl_reset = self.cl_target, 
                           efficiency_param = float(_find_values(self.log_path, 'efficiency_param')),
                           cl_wide = float(_find_values(self.log_path, 'cl_wide')),
                           render_mode="human",
                           n_boxes=1,
                           reynolds = self.reynolds,
                           boxes = self.boxes)
            self.model = PPO.load(self.model_path, env=self.env)

            if self.logs >= 2:
                print("*** Model loaded in", time.time()-start_time, "seconds")

        elif model == 'twobox':
            # Obtiene la ruta del directorio donde se encuentra el script actual
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Construye la ruta al modelo de manera relativa al script
            self.model_path = os.path.join(current_dir, "models", "twobox", "twobox.zip")
            #self.model_path = "drlfoil/models/onebox/onebox.zip"
            #self.log_path = "drlfoil/models/onebox/log_onebox.txt"
            self.log_path = os.path.join(current_dir, "models", "twobox", "log_twobox.txt")

            if self.logs >= 1:
                print("*** Loading model from", self.model_path, "***")

            if len(self.boxes) != 2:
                raise ValueError("Twobox model requires two box restrictions")
            
            start_time = time.time()
            # Load the environment with the parameters from the log file and cl targey & reynolds defined by the user
            self.env = gym.make('AirfoilEnv-v0', 
                           n_params= int(_find_values(self.log_path, 'n_params')),
                           max_steps= steps,
                           scale_actions = float(_find_values(self.log_path, 'scale_actions')),
                           airfoil_seed = [0.1*np.ones(int(_find_values(self.log_path, 'n_params'))), -0.1*np.ones(int(_find_values(self.log_path, 'n_params'))), 0.0], #TO BE CHANGED
                           delta_reward= False, 
                           cl_reward = bool(_find_values(self.log_path, 'cl_reward')),
                           cl_reset = self.cl_target, 
                           efficiency_param = float(_find_values(self.log_path, 'efficiency_param')),
                           cl_wide = float(_find_values(self.log_path, 'cl_wide')),
                           render_mode="human",
                           n_boxes=2,
                           reynolds = self.reynolds,
                           boxes = self.boxes)
            self.model = PPO.load(self.model_path, env=self.env)
            if self.logs >= 2:
                print("*** Model loaded in", time.time()-start_time, "seconds")

        elif model == 'nobox':
            # Obtiene la ruta del directorio donde se encuentra el script actual
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Construye la ruta al modelo de manera relativa al script
            self.model_path = os.path.join(current_dir, "models", "nobox", "nobox.zip")
            #self.model_path = "drlfoil/models/onebox/onebox.zip"
            #self.log_path = "drlfoil/models/onebox/log_onebox.txt"
            self.log_path = os.path.join(current_dir, "models", "nobox", "log_nobox.txt")

            if self.logs >= 1:
                print("*** Loading model from", self.model_path, "***")

            if len(self.boxes) != 0:
                raise ValueError("Nobox model requires no box restrictions")

            start_time = time.time()
            # Load the environment with the parameters from the log file and cl targey & reynolds defined by the user
            self.env = gym.make('AirfoilEnv-v0', 
                           n_params= int(_find_values(self.log_path, 'n_params')),
                           max_steps= steps,
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
        self.bestairfoil = None # Placeholder for the best airfoil found (AirfoilTools object)
        self.bestairfoil_reward = -np.inf
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.env.step(action)

            if self.logs == 3:
                print("*** Airfoil found with an efficiency of", info['efficiency'], "and lift coefficient of", info['cl'])

            if reward > self.bestairfoil_reward:
                self.bestairfoil = copy.deepcopy(self.env.unwrapped.state)
                self.bestairfoil_reward = reward

                if self.logs == 3:
                    print("*** New best airfoil found!")

        if self.logs >= 2:
            print("*** Optimization finished! Time elapsed:", time.time()-start_time, "seconds")
        if self.logs >= 1:
            print(f"***Best airfoil found with a lift coefficient of {self.bestairfoil.get_cl()} (target: {self.cl_target}) and efficiency of {self.bestairfoil.get_efficiency()}")   
            self.bestairfoil.airfoil_plot()

    def save(self, name : str):
        """
        Function used to save the best airfoil found in a .dat file with the name given by the user

        Args:
        -name: Name of the file
        """

        airfoil_coords = self.bestairfoil.get_coordinates()
        with open(f"{name}.dat", 'w') as f:
            f.write(f"{name}\n")

            for item in range(len(airfoil_coords[0])):
                # Pass the first element since it is the same as the last element of the upper surface
                f.write(str(airfoil_coords[0][item][0])+ '   '+str(airfoil_coords[0][item][1])+ '\n')

            for item in range(len(airfoil_coords[1])):
                # Pass the first element since it is the same as the last element of the upper surface
                if item == 0:
                    continue
                f.write(str(airfoil_coords[1][item][0])+ '   '+str(airfoil_coords[1][item][1])+ '\n')


    def reset(self, reynolds = None, cl_target = None) -> None:
        """
        Used to reset the environment with new parameters. If no parameters are given, the environment is reset with the same parameters as before.

        Args:
        -reynolds: Reynolds number of the flow
        -cl_target: Target lift coefficient
        """

        options = {}
        if reynolds is not None:
            options['reynolds'] = reynolds
            self.reynolds = reynolds
        if cl_target is not None:
            options['cl_target'] = cl_target
            self.cl_target = cl_target

        self.env.reset(options=options)
        self.bestairfoil = None

    def show(self,) -> None:
        """
        Plot the airfoil found
        """

        if self.bestairfoil is None:
            raise ValueError("No airfoil found yet. You have to run the optimization first. If you want analyze a different airfoil, check drfoil.utilities")
        self.bestairfoil.airfoil_plot()

    def analyze(self, plot : bool = False) -> dict:
        """
        Analyze the airfoil found using Neuralfoil. It returns a dictionary with the lift coefficient, drag coefficient and efficiency for each angle of attack.

        Args:
        -plot: If True, it plots the lift coefficient, drag coefficient and efficiency against the angle of attack
        """
        if self.bestairfoil is None:
            raise ValueError("No airfoil found yet. You have to run the optimization first. If you want analyze a different airfoil, check drfoil.utilities")
        return AeroAnalysis(self.bestairfoil, reynolds=self.reynolds, plot=plot)

    