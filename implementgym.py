
import gym
import environment
import numpy as np

# Create the environment
env = gym.make('AirfoilEnv-v0', n_params=15)

# Reset the environment

observation, _ = env.reset()


observationn, _, _, _, _ = env.step(action=[0.0*np.ones(15, dtype= np.float32), 0.0*np.ones(15, dtype= np.float32), np.array([0.0], dtype=np.float32)])
print(observationn)
print(observation)