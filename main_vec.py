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

today = date.today()
formatted_date = today.strftime("%d%m%y")

# tensorboard --logdir .\logs\tensorboard_logs\FECHA\MODELO

############################### MODEL NAME ########################################
MODEL_NAME = f"{formatted_date}_ClReward_NoDelta_Wide20"
############################### MODEL NAME ########################################

LOG_DIR = f"./logs/eval_logs/{formatted_date}/{MODEL_NAME}"
TENSORBOARD_DIR = f"./logs/tensorboard_logs/{formatted_date}/"


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make('AirfoilEnv-v0', n_params=10, max_steps=10, scale_actions = 0.35, airfoil_seed = [0.1*np.ones(10), -0.1*np.ones(10), 0.0],
                       delta_reward=False, cl_reward = True, cl_reset = None, efficiency_param = 1, cl_wide = 20)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init



if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)


    env_id = 'AirfoilEnv-v0'
    num_cpu = 10  # Number of processes to use


    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Reset the environment
    vec_env.reset()
    #env.render()

    eval_callback = EvalCallback(vec_env, best_model_save_path=LOG_DIR,
                                log_path=LOG_DIR, eval_freq=max(2000, 1),
                                n_eval_episodes=5, deterministic=False,
                                render=False)


    # Instantiate the agent
    model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=TENSORBOARD_DIR)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2000000), progress_bar=True, callback=eval_callback, tb_log_name=MODEL_NAME)
    # Save the agent
    model.save(MODEL_NAME)
    #del model  # delete trained model to demonstrate loading"