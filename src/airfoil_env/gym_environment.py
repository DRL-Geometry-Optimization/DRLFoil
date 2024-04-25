import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

from .parametrization import AirfoilTools
from .reward import reward
from .restriction import BoxRestriction

# Tutorial: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/


class AirfoilEnv(gym.Env):
    """
    Airfoil environment for reinforcement learning. 
    The environment is based on the AirfoilTools class from the parametrization module.
    """


    metadata = {'render_modes': ["human", "no_display"], "render_fps": 2 }

    _BOX_LIMIT = 1 # Maximum number of boxes in the airfoil

    def __init__(self, render_mode : bool = None, max_steps : int = 10, reward_threshold : bool = None, # Environment parameters
                 n_params : int = 10, scale_actions : float = 0.15, airfoil_seed : np.ndarray = [0.1*np.ones(10), -0.1*np.ones(10), 0.0], # Initial state of the environment
                 cl_reward : bool = True, cl_reset : float = None, cl_wide : float = 20, # Cl reward parameters
                 delta_reward : bool = False, # Activate the delta reward
                 efficiency_param : float = 1, # Efficiency weight parameter
                 n_boxes : int = 1,): # Number of boxes in the airfoil
        
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

        self.cl_wide = cl_wide
        self.delta_reward = delta_reward
        self.efficiency_param = efficiency_param
        self.scale_actions = scale_actions
        self.airfoil_seed = airfoil_seed

        # Create the airfoil object
        self.state = AirfoilTools() 
        self.n_params = n_params # Number of parameters in one side of the airfoil

        # Initialize the environment state
        self.done = False 
        self.step_counter = 0
        self.reward = 0
        self.last_efficiency = None # Placeholder for the last efficiency value

        self.render_mode = render_mode

        if n_boxes > self._BOX_LIMIT:
            raise ValueError(f"The number of boxes is limited to {self._BOX_LIMIT}")
        else:
            self.n_boxes = n_boxes


        # Spaces dict is not used since it means observations are from different types of data. MultiLayerInput 
        # of Stable Baselines 3 is not the most efficient way to handle this. 

        obs_dict = {} # Placeholder for the observation space
        obs_dict["airfoil"] = spaces.Box(low=-5.0, high=5.0, shape=(2*self.n_params + 1,), dtype=np.float32)

        if self.cl_reward == True:
            obs_dict["cl_target"] = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

        if self.n_boxes > 0:
            obs_dict["boxes"] = spaces.Box(low=-5.0, high=5.0, shape=(4*self.n_boxes,), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)


        """if cl_reward == True:
            self.observation_space = spaces.Dict({
                "airfoil": spaces.Box(low=-5.0, high=5.0, shape=(2*self.n_params + 1,), dtype=np.float32),
                "boxes": spaces.Box(low=-2.0, high=2.0, shape=(4*self._BOX_LIMIT,), dtype=np.float32),
                "cl_target": spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Dict({
                "airfoil": spaces.Box(low=-5.0, high=5.0, shape=(2*self.n_params + 1,), dtype=np.float32),
                "boxes": spaces.Box(low=-5.0, high=5.0, shape=(4*self._BOX_LIMIT,), dtype=np.float32),
            })"""


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2*self.n_params+1,), dtype=np.float32)



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
            raise NotImplementedError("Due to a mistake, it is necessary to implement the airfoil_seed parameter in the reset method. random kulfan needs an airfoil declaration first")
            self.state.random_kulfan2(n_params= self.n_params)


        # Box reset
        self.state.boxes = [] # Reset the boxes

        # Box creation
        if self.n_boxes > self._BOX_LIMIT:
            raise ValueError(f"The number of boxes is limited to {self._BOX_LIMIT}")
        
        if self.n_boxes == 0:
            pass
        elif self.n_boxes == 1:
            self.state.get_boxes(BoxRestriction.random_box(y_simmetrical=False, ymin=-0.05, ymax=0.10, widthmax=0.55, heightmax=0.08))
        elif self.n_boxes == 2:
            self.state.get_boxes(BoxRestriction.random_box(y_simmetrical=False, ymin=-0.05, ymax=0.10, widthmax=0.55, heightmax=0.08,
                                                           xmax=0.5))
            self.state.get_boxes(BoxRestriction.random_box(y_simmetrical=False, ymin=-0.05, ymax=0.10, widthmax=0.55, heightmax=0.08,
                                                           xmin=0.5))



        self.done = False
        self.step_counter = 0

        if self.cl_reward == True and self.cl_reset is None:
            self.cl_target = random.uniform(0.1, 1.2)


        upper, lower, le = self.state.get_weights()


        observation = {"airfoil": np.array(upper + lower + le, dtype=np.float32),}

        if self.n_boxes > 0:
            boxes_obs = np.zeros(4*self.n_boxes, dtype=np.float32)

            for i in range(self.n_boxes):
                boxes_obs[4*i:4*(i+1)] = self.state.return_boxes()[i]

            observation["boxes"] = boxes_obs


        if self.cl_reward == True:
            observation["cl_target"] = np.array([self.cl_target], dtype=np.float32)



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
        

        observation = {"airfoil": np.array(upper + lower + le, dtype=np.float32),}

        if self.cl_reward == True:
            observation["cl_target"] = np.array([self.cl_target], dtype=np.float32)


        if self.n_boxes > 0:
            boxes_obs = np.zeros(4*self.n_boxes, dtype=np.float32)

            for i in range(self.n_boxes):
                boxes_obs[4*i:4*(i+1)] = self.state.return_boxes()[i]

            observation["boxes"] = boxes_obs


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