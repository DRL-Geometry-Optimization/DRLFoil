import numpy as np
import gym
from gym import spaces
#import gymnasium as gym
#from gymnasium import spaces

from .parametrization import airfoiltools
from .reward import reward

# Tutorial: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/


class AirfoilEnv(gym.Env):
    metadata = {'render.modes': ["human", "no_display"], "render_fps": 4 }

    def __init__(self, n_params = 15, # Initial state of the environment
                 max_steps=50, reward_threshold=None, # Iterations control
                 cl_reward=False, cl_target=None, cl_maxreward=40, cl_wide=15, delta_reward=False, efficiency_param=1): # Reward control

        # state0 should have the following structure: [[UPPARAMETERS],[DOWNPARAMETERS],LE_weight]

        # NOTE: IF CL_TARGET IS ACTIVATED, THE NEURAL NETWORK SHOULD HAVE THE CL TARGET AS INPUT
        # NOTE: s0 SHOULD BE USED ON RESET METHOD TO RESET THE ENVIRONMENT TO THE INITIAL STATE

        
        # Input parameters
        self.max_steps = max_steps
        self.efficiency_th = reward_threshold
        self.cl_reward = cl_reward
        self.cl_target = cl_target
        self.cl_maxreward = cl_maxreward
        self.cl_wide = cl_wide
        self.delta_reward = delta_reward
        self.efficiency_param = efficiency_param

        # Create the airfoil object
        self.state = airfoiltools() # Create an airfoil object
        self.n_params = n_params # Number of parameters in one side of the airfoil

        # Initialize the environment state
        self.done = False # The episode is not done by default 
        self.step_counter = 0
        self.reward = 0
        self.last_efficiency = None # Placeholder for the last efficiency value



        higher_action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32)
        lower_action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32)
        le_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.action_space = spaces.Tuple((higher_action_space, lower_action_space, le_action_space))
        
        # The observation space is the airfoil
        higher_obs_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_params,), dtype=np.float32)
        lower_obs_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_params,), dtype=np.float32)
        le_obs_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Tuple((higher_obs_space, lower_obs_space, le_obs_space))

        #self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(2*self.n_params+1,), dtype=np.float32)


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

        upper, lower, le = self.state.get_weights()
        #observation = np.array(upper + lower+ le, dtype=np.float32)
        observation = np.array(upper, dtype=np.float32), np.array(lower, dtype=np.float32), np.array(le, dtype=np.float32)

        info = {}
        self.state.airfoil_plot()

        return observation, info



    def step(self, action):
        """
        This method takes an action and returns the new state, the reward, and whether the episode is done.
        """

        if self.step_counter == 0:
            self.state.analysis()
            self.last_efficiency = self.state.get_efficiency()


        # Update the state of the environment
        self.state.modify_airfoil(action)

        # Reward calculation
        if self.state.check_airfoil() == False:
            self.reward = -100
            # Last efficiency is not updated
        else:
            self.state.analysis() # Analyze the airfoil

            # NOTE: REWARD SHOULD BE UPTADETED TO INCLUDE THE CL TARGET
            self.reward = reward(efficiency=self.state.get_efficiency(),
                                efficiency_param=self.efficiency_param, 
                                last_efficiency=self.last_efficiency,
                                cl_reward=self.cl_reward,
                                cl=self.state.get_cl(),
                                cl_target=self.cl_target, 
                                cl_maxreward=self.cl_maxreward, 
                                cl_wide=self.cl_wide, 
                                delta_reward=self.delta_reward)
            # Update the last efficiency
            self.last_efficiency = self.state.get_efficiency()

        self.step_counter += 1

        # Check if the episode is done
        if self.efficiency_th is not None:
            if self.state.get_efficiency() >= self.efficiency_th:
                self.done = True

        if self.step_counter > self.max_steps:
            self.done = True

        upper, lower, le = self.state.get_weights()
        #observation = np.array(upper + lower+ le, dtype=np.float32)
        observation = np.array(upper, dtype=np.float32), np.array(lower, dtype=np.float32), np.array(le, dtype=np.float32)

        return observation, self.reward, self.done, False, {}
    

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