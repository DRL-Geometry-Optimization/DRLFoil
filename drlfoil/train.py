from drlfoil.recorder import CreateLog

import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from datetime import date
import torch.nn as nn


def _make_env(env_id: str, rank: int, n_params: int, max_steps: int, 
             scale_actions: float, airfoil_seed: list, delta_reward: bool, 
             cl_reward: bool, cl_reset: float, efficiency_param: float, 
             cl_wide: float, n_boxes: int, reynolds: int, seed = 0):
    """
    Factory function to create an instance of the Gym environment with specified parameters.
    This function is defined outside of the Train class to avoid serialization issues
    with Python's multiprocessing module.
    """


    def _init():
        env = gym.make(env_id, 
                       n_params=n_params, 
                       max_steps=max_steps, 
                       scale_actions=scale_actions, 
                       airfoil_seed=airfoil_seed,
                       delta_reward=delta_reward, 
                       cl_reward=cl_reward, 
                       cl_reset=cl_reset, 
                       efficiency_param=efficiency_param, 
                       cl_wide=cl_wide,
                       n_boxes=n_boxes,
                       reynolds=reynolds)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class Train:
    def __init__(self, modelname : str, name_date : bool = False):
        
        self.modelname = modelname
        if name_date:
            today = date.today()
            formatted_date = today.strftime("%d%m%y")
            self.modelname = self.modelname + "_" + formatted_date


        self.vec_env = None
        self.test_env = None
        self.model = None


        # Environment parameters
        self.n_params = None
        self.n_boxes = None
        self.reynolds = None
        self.max_steps = None
        self.scale_actions = None
        self.airfoil_seed = None
        self.delta_reward = None
        self.cl_reward = None
        self.cl_reset = None
        self.efficiency_param = None
        self.cl_wide = None

        # Training parameters
        self.total_timesteps = None
        self.num_cpu = None
        self.test_num_cpu = None
        self.env_id = 'AirfoilEnv-v0'

        # Model parameters
        self.net_arch = None
        self.activation_fn = None
        self.gamma = None
        self.learning_rate = None
        self.ent_coef = None
        self.batch_size = None
        self.clip_range = None
        self.gae_lambda = None
        self.max_grad_norm = None
        self.n_epochs = None
        self.n_steps = None
        self.vf_coef = None

    def environment_parameters(self, n_boxes : int, n_params : int = 8, reynolds : int = None, 
                                   max_steps : int = 10, scale_actions : float = 0.15, 
                                   airfoil_seed : list = [0.1*np.ones(8), -0.1*np.ones(8), 0.0], 
                                   delta_reward : bool = False, cl_reward : bool = True, cl_reset : float = None, 
                                   efficiency_param : float = 1, cl_wide : float = 20):
        self.n_params = n_params
        self.n_boxes = n_boxes
        self.reynolds = reynolds
        self.max_steps = max_steps
        self.scale_actions = scale_actions
        self.airfoil_seed = airfoil_seed
        self.delta_reward = delta_reward
        self.cl_reward = cl_reward
        self.cl_reset = cl_reset
        self.efficiency_param = efficiency_param
        self.cl_wide = cl_wide

    def training_parameters(self, total_timesteps : int = 3500000, num_cpu : int = 48, test_num_cpu : int = 1):
        self.total_timesteps = total_timesteps
        self.num_cpu = num_cpu
        self.test_num_cpu = test_num_cpu

    def model_parameters(self, net_arch : list = [256,256,256], activation_fn : callable = nn.Tanh, 
                             gamma : float = 0.995, learning_rate : float = 0.000268, ent_coef : float = 0.0, 
                             batch_size : int = 512, clip_range : float = 0.3, gae_lambda : float = 0.98,
                             max_grad_norm : float = 5.0, n_epochs : int = 20, n_steps : int = 32, 
                             vf_coef : float = 0.754843):
            self.net_arch = net_arch
            self.activation_fn = activation_fn
            self.gamma = gamma
            self.learning_rate = learning_rate
            self.ent_coef = ent_coef
            self.batch_size = batch_size
            self.clip_range = clip_range
            self.gae_lambda = gae_lambda
            self.max_grad_norm = max_grad_norm
            self.n_epochs = n_epochs
            self.n_steps = n_steps
            self.vf_coef = vf_coef

    def copy_model_parameters(self):
         """
         Copy the parameters from a DRLFoil log file
         """
         pass    


    def train(self):
        self.vec_env = SubprocVecEnv([_make_env(self.env_id, i, self.n_params, self.max_steps, 
                                                 self.scale_actions, self.airfoil_seed, self.delta_reward, 
                                                 self.cl_reward, self.cl_reset, self.efficiency_param, 
                                                 self.cl_wide, self.n_boxes, self.reynolds) 
                                       for i in range(self.num_cpu)])
        self.test_env = SubprocVecEnv([_make_env(self.env_id, i, self.n_params, self.max_steps,
                                                 self.scale_actions, self.airfoil_seed, self.delta_reward, 
                                                 self.cl_reward, self.cl_reset, self.efficiency_param, 
                                                 self.cl_wide, self.n_boxes, self.reynolds) 
                                       for i in range(self.test_num_cpu)])
         
        self.model = PPO("MultiInputPolicy", self.vec_env, verbose=1,
                         gamma=self.gamma, learning_rate=self.learning_rate, ent_coef=self.ent_coef, 
                         batch_size=self.batch_size, clip_range=self.clip_range, gae_lambda=self.gae_lambda, 
                         max_grad_norm=self.max_grad_norm, n_epochs=self.n_epochs, n_steps=self.n_steps, 
                         vf_coef=self.vf_coef, device='cuda', policy_kwargs=dict(net_arch=self.net_arch, 
                                                                              activation_fn=self.activation_fn))
        
        self.model.learn(total_timesteps=self.total_timesteps, progress_bar=True)