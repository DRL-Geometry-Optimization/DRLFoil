import gym
import environment
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy




# Create the environment
env = gym.make('AirfoilEnv-v0', n_params=10, max_steps=20)
# Reset the environment
env.reset()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(60000), progress_bar=True)
# Save the agent
model.save("TestModel_11042024_1")
del model  # delete trained model to demonstrate loading