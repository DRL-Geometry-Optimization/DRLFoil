
import gym
import environment
import numpy as np

# Create the environment
env = gym.make('AirfoilEnv-v0', n_params=15, max_steps=3)

# Reset the environment

observation, _ = env.reset()

for i in range(6):
    observationn, reward, done, _, xd = env.step(action=0.7*np.ones(30))
    print("STEP:", xd)
    print("STATE:", observationn)
    print("REWARD:", reward)
    print("DONE:", done)
    print("")
    if done:
        break
