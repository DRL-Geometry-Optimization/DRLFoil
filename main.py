import gym
import environment
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy




# Create the environment
env = gym.make('AirfoilEnv-v0', n_params=10, max_steps=10, scale_actions = 0.4, airfoil_seed = [0.1*np.ones(10), -0.1*np.ones(10), 0.0])
# Reset the environment
env.reset()
env.render()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[128, 128]))
# Train the agent and display a progress bar
model.learn(total_timesteps=int(200000), progress_bar=True)
# Save the agent
model.save("TestModel_14042024_4_seedairfoil_net128")
#del model  # delete trained model to demonstrate loading"