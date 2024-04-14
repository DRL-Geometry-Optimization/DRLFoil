import environment
from environment.parametrization import airfoiltools

import gym
import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import EvalCallback

class ModelManagement():
    def __init__(self, model, env) -> None:
        self.model = model
        self.env = env
        self.airfoil = None

        def train(self, total_timesteps = 200000, progress_bar = True, callback = None, save_path = None):
            self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar, callback=callback)

        def get_airfoil(self, airfoil):
            self.airfoil = airfoiltools()
            self.airfoil.kulfan(airfoil[0], airfoil[1], airfoil[2], airfoil[3], airfoil[4])
    
