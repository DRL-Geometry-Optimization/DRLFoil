from environment.gym_environment import AirfoilEnv
from environment.parametrization import airfoiltools
import numpy as np

# Create the environment
env = AirfoilEnv(state0=[0.1*np.ones(10), 0.2*np.ones(10), 0.1])

#print(env.state.get_weights())

print(env.reset())