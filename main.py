import gymnasium as gym
import environment
import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import EvalCallback

eval_log_dir = "./eval_logs/"
os.makedirs(eval_log_dir, exist_ok=True)


env_id = 'AirfoilEnv-v0'
#n_training_envs = 1
#n_eval_envs = 5


# Create the environment
env = gym.make('AirfoilEnv-v0', n_params=10, max_steps=10, scale_actions = 0.4, airfoil_seed = [0.1*np.ones(10), -0.1*np.ones(10), 0.0],
               delta_reward=True)
#train_env = make_vec_env(env_id, n_envs=n_training_envs, n_params=10, max_steps=10, scale_actions = 0.4, airfoil_seed = [0.1*np.ones(10), -0.1*np.ones(10), 0.0])

# Reset the environment
env.reset()
env.render()

eval_callback = EvalCallback(env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=max(2000, 1),
                              n_eval_episodes=5, deterministic=False,
                              render=False)


# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[128, 128]))
# Train the agent and display a progress bar
model.learn(total_timesteps=int(400000), progress_bar=True, callback=eval_callback)
# Save the agent
model.save("TITULO")
#del model  # delete trained model to demonstrate loading"