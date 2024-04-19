import gymnasium as gym
import environment
import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.utils import set_random_seed


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make('AirfoilEnv-v0', n_params=10, max_steps=20, scale_actions = 0.25, airfoil_seed = [0.1*np.ones(10), -0.1*np.ones(10), 0.0],
                       delta_reward=False, cl_reward = True, cl_reset = None, efficiency_param = 1, cl_wide = 20)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init



if __name__ == "__main__":
    eval_log_dir = "./eval_logs/"
    os.makedirs(eval_log_dir, exist_ok=True)


    env_id = 'AirfoilEnv-v0'
    num_cpu = 20  # Number of processes to use


    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Reset the environment
    vec_env.reset()
    #env.render()

    eval_callback = EvalCallback(vec_env, best_model_save_path=eval_log_dir,
                                log_path=eval_log_dir, eval_freq=max(2000, 1),
                                n_eval_episodes=5, deterministic=False,
                                render=False)


    # Instantiate the agent
    model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=dict(net_arch=[512, 512, 256]))
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2000000), progress_bar=True, callback=eval_callback)
    # Save the agent
    model.save("19042024_Restriction_1_512_512_256")
    #del model  # delete trained model to demonstrate loading"