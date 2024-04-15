import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

from .parametrization import airfoiltools
from .reward import reward

# Tutorial: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/


class AirfoilEnv(gym.Env):
    """
    Airfoil environment for reinforcement learning. 
    The environment is based on the airfoiltools class from the parametrization module.
    """

    metadata = {'render_modes': ["human", "no_display"], "render_fps": 2 }

    def __init__(self, render_mode : bool = None, max_steps : int = 50, reward_threshold : bool = None, # Environment parameters
                 n_params : int = 15, scale_actions : float = 1, airfoil_seed : np.ndarray = None, # Initial state of the environment
                 cl_reward : bool = False, cl_reset : float = None, cl_maxreward : float = 80, cl_wide : float = 15, # Cl reward parameters
                 delta_reward : bool = False, # Activate the delta reward
                 efficiency_param : float = 1): # Efficiency weight parameter
        
        """
        Initialize the environment with the following parameters:
        - render_mode: The mode of rendering the environment. It can be "human" or "no_display".
        - max_steps: The maximum number of steps in the environment.
        - reward_threshold: The threshold for the reward. If the efficiency is above this threshold, the episode is done.
        - n_params: The number of parameters in one side of the airfoil.
        - scale_actions: The scaling factor for the actions. It is used because action space is normalized between -1 and 1.
        - airfoil_seed: The initial seed for the airfoil. If it is None, the airfoil is randomly generated.
        airfoil seed should have the following structure: [upper_weights, lower_weights, leading_edge_weight]
        - cl_reward: If True, the reward is also based on the Cl value of the airfoil.
        - cl_reset: The fixed Cl target for the airfoil. If it is None, the Cl target is randomly generated on every reset.
        - cl_maxreward: The maximum reward for the Cl bell function.
        - cl_wide: The width of the bell function for the Cl reward.
        - delta_reward: If True, the reward is based on the difference between the current efficiency and the last efficiency.
        - efficiency_param: The weight of the efficiency in the reward function.
        """

        
        # Input parameters
        self.max_steps = max_steps
        self.efficiency_th = reward_threshold
        self.cl_reward = cl_reward

        self.cl_reset = cl_reset

        if cl_reward == True and self.cl_reset is not None:
            self.cl_target = self.cl_reset # Cl target is fixed
        else:
            self.cl_target = None # Placeholder for the cl target that will be randomly generated in the reset method

        self.cl_maxreward = cl_maxreward
        self.cl_wide = cl_wide
        self.delta_reward = delta_reward
        self.efficiency_param = efficiency_param
        self.scale_actions = scale_actions
        self.airfoil_seed = airfoil_seed

        # Create the airfoil object
        self.state = airfoiltools() 
        self.n_params = n_params # Number of parameters in one side of the airfoil

        # Initialize the environment state
        self.done = False 
        self.step_counter = 0
        self.reward = 0
        self.last_efficiency = None # Placeholder for the last efficiency value

        self.render_mode = render_mode


        # Spaces dict is not used since it means observations are from different types of data. MultiLayerInput 
        # of Stable Baselines 3 is not the most efficient way to handle this. 
        """self.observation_space = spaces.Dict({
            "upper": spaces.Box(low=-5.0, high=5.0, shape=(self.n_params,), dtype=np.float32),
            "lower": spaces.Box(low=-5.0, high=5.0, shape=(self.n_params,), dtype=np.float32),
            "le": spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        })"""

        """self.action_space = spaces.Dict({
            "upper": spaces.Box(low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32),
            "lower": spaces.Box(low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32),
            "le": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })"""

        # Actions: 
        # 1. Upper side parameters
        # 2. Lower side parameters
        # 3. Leading edge weight
        # 4. Cl target (if activated)

        # space is the weights of the airfoil, the leading edge weight and the cl target (if activated)
        if cl_reward == True:
            space = 2*self.n_params + 2
        else:
            space = 2*self.n_params + 1

        # The actions will be everytime the weights of the airfoil. Cl target is not going to be modified
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2*self.n_params+1,), dtype=np.float32)
        # The observations will be the weights of the airfoil, the leading edge weight and the cl target (if activated)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(space,), dtype=np.float32)



    def _get_info(self) -> dict:
        """
        This method returns additional information about the environment state.
        """
        return {"CL": self.state.get_cl, "efficiency": self.state.get_efficiency, "step": self.step_counter}


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        # Reset the environment state 
        if self.airfoil_seed is not None:
            self.state.kulfan(upper_weights=self.airfoil_seed[0], lower_weights=self.airfoil_seed[1], leading_edge_weight=self.airfoil_seed[2])
        else:
            self.state.random_kulfan2(n_params= self.n_params)

        self.done = False
        self.step_counter = 0

        if self.cl_reward == True and self.cl_reset is None:
            self.cl_target = random.uniform(0.1, 1.1)


        upper, lower, le = self.state.get_weights()

        if self.cl_reward == True:
            observation = np.array(upper + lower + le + [self.cl_target], dtype=np.float32)
        else:
            observation = np.array(upper + lower + le, dtype=np.float32)


        """observation = {
            "upper": np.array(upper, dtype=np.float32),
            "lower": np.array(lower, dtype=np.float32),
            "le": np.array(le, dtype=np.float32)
        }"""

        self.state.analysis() # Analyze the airfoil
        self.last_efficiency = self.state.get_efficiency()

        info = {} # Placeholder for additional information

        #self.state.airfoil_plot() # Plot the airfoil

        return observation, info



    def step(self, action : np.ndarray):

        # Scale the action
        action = action * self.scale_actions


        # Update the state of the environment
        self.state.modify_airfoil(action)

        # Reward calculation
        if self.state.check_airfoil() == False: # If the airfoil is invalid (upper side is below the lower side)
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
            # Since last efficiency is going to be updated, the last efficiency output is saved
            #last_efficiency_output = self.last_efficiency
            # Update the last efficiency
            self.last_efficiency = self.state.get_efficiency()

        self.step_counter += 1

        # Check if the episode is done
        if self.efficiency_th is not None:
            if self.state.get_efficiency() >= self.efficiency_th:
                self.done = True

        if self.step_counter >= self.max_steps:
            self.done = True

        upper, lower, le = self.state.get_weights()
        

        if self.cl_reward == True:
            observation = np.array(upper + lower + le + [self.cl_target], dtype=np.float32)
        else:
            observation = np.array(upper + lower + le, dtype=np.float32)

        #self.state.airfoil_plot() # Plot the airfoil

        return observation, self.reward, self.done, False, {"step": self.step_counter, "efficiency": self.state.get_efficiency(),
                                                            "cl": self.state.get_cl()}
    

    def render(self):
        if self.render_mode == "human":
            self.state.airfoil_plot()
            
        elif self.render_mode == "no_display":
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