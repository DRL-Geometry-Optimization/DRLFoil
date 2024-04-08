import numpy as np
import gymnasium as gym
from gymnasium import spaces

from parametrization import airfoiltools
from reward import reward

# Tutorial: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/


class AirfoilEnv(gym.Env):
    metadata = {'render.modes': ["human", "no_display"], "render_fps": 4 }

    def __init__(self, state0, max_iter=300, efficiency_th=None):

        # state0 should have the following structure: [[downparameters],[upparameters],LE_weight]

        if len(state0) == 3:
            pass
        else:
            raise ValueError(f"State should be an array of 3 arrays. Length obtained: {len(state0)}")

        #self.upparameters = len(state0[0]) # Number of parameters in the upper side of the airfoil
        #self.downparameters = len(state0[1])
        self.max_iter = max_iter
        self.efficiency_th = efficiency_th 
        self.state = airfoiltools() # Create an airfoil object
        self.state.kulfan(state0[0], state0[1], state0[2]) # Create the airfoil with the Kulfan parameterization
        self.n_params = self.state.numparams # Number of parameters in one side of the airfoil
        self.done = False # The episode is not done by default 
        self.step_counter = 0
        self.reward = 0

        self.last_efficiency = None # Placeholder for the last efficiency value

        # The action space is the weights of the airfoil. +1 is for the weight of the leading edge
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_params+1,), dtype=np.float32) 
        
        # The observation space is the airfoil
        self.observation_space = spaces.Box(low=np.NINF, high=np.Inf, shape=(self.n_params+1,), dtype=np.float32)


    def _get_info(self):
        # Return more information about the environment state (for debugging)
        return {"CL": self.state.get_cl, "efficiency": self.state.get_efficiency, "step": self.step_counter}


    def reset(self, seed=None, options=None):
        """
        This method resets the environment to the initial state.
        """

        super().reset(seed=seed)
        self.state.random_kulfan2(n_params= self.n_params)

        self.done = False
        self.step_counter = 0

        observation = self.state.get_weights()

        return observation



    def step(self, action):
        """
        This method takes an action and returns the new state, the reward, and whether the episode is done.
        """

        if self.step_counter == 0:
            self.state.analysis()
            self.last_efficiency = self.state.get_efficiency()


        # Update the state of the environment
        self.state.modify_airfoil(action)
        self.state.analysis() # Analyze the airfoil


        self.reward = reward(self.state.get_efficiency(), self.last_efficiency)
        self.last_efficiency = self.state.get_efficiency()

        self.step_counter += 1

        # Check if the episode is done
        if self.step_counter >= self.max_iter:
            self.done = True

        # NOTE: Falta definir una forma de devolver correctamente el airfoil
        # Normalizar parametrization.py para que trabaje en el formato actual
        return self.state.get_weights(), self.reward, self.done, {}
    

    def render(self):
        pass



if __name__ == "__main__":

    pedroduque = AirfoilEnv(state0=[0.1*np.ones(10), 0.2*np.ones(10), 0.1])

    observation = pedroduque.reset()
    observationn, reward, done, info = pedroduque.step(action=[0.1*np.ones(10), 0.2*np.ones(10), 0.1])
    print(observationn)

    """
    # Create an instance of the AirfoilEnv class
    env = AirfoilEnv(state0=[0.1*np.ones(10), 0.2*np.ones(10), 0.1], max_iter=20, efficiency_th=0.8)

    # Reset the environment
    observation = env.reset()

    # Run the simulation for a fixed number of steps
    for _ in range(2):
        # Choose a random action
        action = env.action_space.sample()

        # Take a step in the environment
        observation, reward, done, info = env.step(action)

        # Print the current state and reward
        print("Observation:", observation)
        print("Reward:", reward)

        # Check if the episode is done
        if done:
            break
            
    """