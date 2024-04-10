from environment.parametrization import airfoiltools
from environment.reward import reward
from environment.gym_environment import AirfoilEnv
import numpy as np 

"""
Testing the environment
"""

s0 = [0.3*np.ones(10), -0.05*np.ones(10), 0.1] # Initial state

# ENVIRONMENT CREATION
env = AirfoilEnv(state0=s0, max_steps=3, reward_threshold=100)
"""print(f"Initial state: {env.state.upper_weights, env.state.lower_weights}")
a, b = env.state.get_coordinates()
print(a[1], b[len(a)-2])
if a[15][0] != b[len(a)-16][0]:
    print("False")"""
env.state.airfoil_plot()

# STEP TESTING
for _ in range(3):
    for i in range(10):
        print(f"Step: {i}")
        action = [-0.1*np.ones(10), 0.05*np.ones(10), 0.1]
        observation, reward, done, info = env.step(action)
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done} (Step {i}), max_steps: {env.max_steps}")
        print(env.state.upper_weights)
        print(f"State: {env.state.upper_weights, env.state.lower_weights}")
        print(f"CL: {env.state.get_cl()}")
        env.state.airfoil_plot()

        if done:
            break

    env.reset()
"""
# REWARD TESTING
efficiency = 20

reward_obtained = reward(efficiency=efficiency, cl_reward=True, cl=0.35, cl_target=0.4, delta_reward=True, last_efficiency=15)

print(f"Reward obtained: {reward_obtained}")
"""


