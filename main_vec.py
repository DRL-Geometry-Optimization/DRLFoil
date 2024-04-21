import gymnasium as gym
import environment
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from datetime import date
from recorder.create_logs import create_log

# tensorboard --logdir .\logs\tensorboard_logs\FECHA\MODELO

# Get the current date
today = date.today()
formatted_date = today.strftime("%d%m%y")

############################### MODEL NAME ########################################
name = "4M_LearningStudy_0.00024_Gamma0.99"
############################### MODEL NAME ########################################



############################ HYPERPARAMETERS #####################################
n_params = 10
max_steps = 10
scale_actions = 0.15
airfoil_seed = [0.1*np.ones(n_params), -0.1*np.ones(n_params), 0.0]
delta_reward = False
cl_reward = True
cl_reset = None
efficiency_param = 1
cl_wide = 20

num_cpu = 12  # Number of processes to use
env_id = 'AirfoilEnv-v0'

net_arch = [512, 512, 256]
total_timesteps = 4000000


gamma = 0.99
learning_rate = 0.00024
############################ HYPERPARAMETERS #####################################




# Model name
MODEL_NAME = f"{formatted_date}_{name}"
# Model directory
MODEL_DIR = f"./models/{formatted_date}/{MODEL_NAME}"
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
                       cl_wide = cl_wide)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init



if __name__ == "__main__":
    # Create log directories
    os.makedirs(MODEL_DIR, exist_ok=False)
    os.makedirs(LOG_DIR, exist_ok=False)

    


    # Create log for the logs directory
    create_log(name=MODEL_NAME, dir=LOG_DIR,
               n_params=n_params, max_steps=max_steps, scale_actions=scale_actions,
               airfoil_seed=airfoil_seed, delta_reward=delta_reward, cl_reward=cl_reward,
               cl_reset=cl_reset, efficiency_param=efficiency_param, cl_wide=cl_wide,
               num_cpu=num_cpu, net_arch=net_arch, total_timesteps=total_timesteps,
               gamma=gamma, learning_rate=learning_rate)

    



    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Reset the environment
    vec_env.reset()
    #env.render()

    eval_callback = EvalCallback(vec_env, best_model_save_path=LOG_DIR,
                                log_path=LOG_DIR, eval_freq=10000,
                                n_eval_episodes=7, deterministic=True,
                                render=False)

    #model = PPO.load("models/210424/210424_LearningStudy_0.00025_Gamma0.99/logs/best_model", env=vec_env, tensorboard_log=MODEL_DIR)

    # Instantiate the agent
    model = PPO("MultiInputPolicy", vec_env, verbose=1, policy_kwargs=dict(net_arch=net_arch), tensorboard_log=MODEL_DIR, 
                gamma=gamma, learning_rate=learning_rate)
    


    # Train the agent and display a progress bar
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback, tb_log_name="TB_LOG")
    # Save the agent
    
    model.save(f"{MODEL_DIR}/{MODEL_NAME}")
    