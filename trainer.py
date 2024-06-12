import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import drlfoil
from drlfoil.recorder import CreateLog

import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from datetime import date
import torch.nn as nn

# tensorboard --logdir .\logs\tensorboard_logs\FECHA\MODELO

# Get the current date
today = date.today()
formatted_date = today.strftime("%d%m%y")

############################### MODEL NAME ########################################
name = "Estudio2BOX_128x128" 
############################### MODEL NAME ########################################

   

############################ HYPERPARAMETERS #####################################
n_params = 8
max_steps = 10
scale_actions = 0.3 
airfoil_seed = [0.3*np.ones(n_params), -0.3*np.ones(n_params), 0.0]
#airfoil_seed = None
delta_reward = False
cl_reward = True
cl_reset = None
efficiency_param = 1
cl_wide = 20


num_cpu = 48  # Number of processes to use
test_num_cpu = 1
env_id = 'AirfoilEnv-v0'

net_arch = [128, 128]
activation_fn = nn.Tanh
total_timesteps = 3000000

n_boxes = 2
reynolds = None

gamma = 0.995
learning_rate = 0.000268
ent_coef = 0.001
batch_size = 512
clip_range = 0.3
gae_lambda = 0.98
max_grad_norm = 5.0
n_epochs = 20
n_steps = 32
vf_coef = 0.754843
############################ HYPERPARAMETERS #####################################




# Model name
MODEL_NAME = f"{formatted_date}_{name}"
# Model directory
MODEL_DIR = f"./logmodels/{formatted_date}/{MODEL_NAME}"
# Log directory
LOG_DIR = f"{MODEL_DIR}/logs"
# Tensorboard directory
#TENSORBOARD_DIR = f"./logs/tensorboard_logs/{formatted_date}/"


# Multiprocessed environment
def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, 
                       n_params=n_params, 
                       max_steps=max_steps, 
                       scale_actions = scale_actions, 
                       airfoil_seed = airfoil_seed,
                       delta_reward= delta_reward, 
                       cl_reward = cl_reward, 
                       cl_reset = cl_reset, 
                       efficiency_param = efficiency_param, 
                       cl_wide = cl_wide,
                       n_boxes=n_boxes,
                       reynolds=reynolds)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init



if __name__ == "__main__":
    # Create log directories
    os.makedirs(MODEL_DIR, exist_ok=False)
    os.makedirs(LOG_DIR, exist_ok=False)

    


    # Create log for the logs directory
    CreateLog(name=MODEL_NAME, dir=LOG_DIR,
               n_params=n_params, max_steps=max_steps, scale_actions=scale_actions,
               airfoil_seed=airfoil_seed, delta_reward=delta_reward, cl_reward=cl_reward,
               cl_reset=cl_reset, efficiency_param=efficiency_param, cl_wide=cl_wide,
               num_cpu=num_cpu, net_arch=net_arch, total_timesteps=total_timesteps,
               gamma=gamma, learning_rate=learning_rate, ent_coef=ent_coef, batch_size=batch_size,
               clip_range=clip_range, gae_lambda=gae_lambda, max_grad_norm=max_grad_norm,
               n_epochs=n_epochs, n_steps=n_steps, vf_coef=vf_coef, activation_fn=activation_fn)

    



    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    test_env = SubprocVecEnv([make_env(env_id, i) for i in range(test_num_cpu)])


    # Reset the environment
    vec_env.reset()
    #env.render()

    eval_callback = EvalCallback(test_env, best_model_save_path=LOG_DIR,
                                log_path=LOG_DIR, eval_freq=50000 // num_cpu,
                                n_eval_episodes=7, deterministic=True,
                                render=False)

    #model = PPO.load("models/210424/210424_LearningStudy_0.00025_Gamma0.99/logs/best_model", env=vec_env, tensorboard_log=MODEL_DIR)

    # Instantiate the agent
    model = PPO("MultiInputPolicy", vec_env, verbose=1, policy_kwargs=dict(net_arch=net_arch, activation_fn = activation_fn), tensorboard_log=MODEL_DIR, 
                gamma=gamma, learning_rate=learning_rate, ent_coef=ent_coef, batch_size=batch_size, clip_range=clip_range,
                gae_lambda=gae_lambda, max_grad_norm=max_grad_norm, n_epochs=n_epochs, n_steps=n_steps, vf_coef=vf_coef,
                device='cuda')
    


    # Train the agent and display a progress bar
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback, tb_log_name="TB_LOG")
    # Save the agent
    
    model.save(f"{MODEL_DIR}/{MODEL_NAME}")
    