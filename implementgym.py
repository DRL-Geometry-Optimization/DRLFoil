
import gym
import environment
import numpy as np

# Create the environment
env = gym.make('AirfoilEnv-v0', state0=[0.1*np.ones(10), 0.2*np.ones(10), 0.1])

# Reset the environment
env.reset()

env.step(action=[0.1*np.ones(10), 0.2*np.ones(10), 0.1])