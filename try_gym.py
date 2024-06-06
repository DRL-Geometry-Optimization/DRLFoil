import sys


import gymnasium as gym
import drlfoil
import numpy as np

# Create the environment

env = gym.make('AirfoilEnv-v0', n_params=8, max_steps=4, scale_actions = 0.15, airfoil_seed = [0.1*np.ones(8), -0.1*np.ones(8), 0.0],
                    delta_reward=False, cl_reward = True, cl_reset = 0.4, efficiency_param = 1, cl_wide = 20, render_mode="human",)

# Reset the environment

observation, _ = env.reset()
print(f"RESET:{observation}")
#observation = env.reset()
env.render()

#action = np.concatenate([0.1*np.ones(15), -0.1*np.ones(15), 0.0])
#print("ACTION:", action)

for i in range(3):
    observationn, reward, done, _, xd = env.step(np.concatenate([0.5*np.ones(15), 0.5*np.ones(15), [0.0]]))
    #observationn, reward, done, _, xd = env.step(action=[0.12*np.ones(15), 0.02*np.ones(15), -0.05])
    print("STEP:", xd)
    print("STATE:", observationn)
    print("REWARD:", reward)
    print("DONE:", done)
    print("")
    env.render()
    if done:
        env.reset()

