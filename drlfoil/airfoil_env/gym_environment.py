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

    _BOX_LIMIT = 2 # Maximum number of boxes in the airfoil
    _CL_MIN = 0.1 # Minimum Cl value for the airfoil
    _CL_MAX = 1.6 # Maximum Cl value for the airfoil
    _RE_MIN = 1e5 # Minimum Reynolds number
    _RE_MAX = 5e7 # Maximum Reynolds number

    def __init__(self, render_mode : bool = None, max_steps : int = 10, reward_threshold : bool = None, # Environment parameters
                 n_params : int = 10, scale_actions : float = 0.15, airfoil_seed : np.ndarray = [0.1*np.ones(10), -0.1*np.ones(10), 0.0], # Initial state of the environment
                 cl_reward : bool = True, cl_reset : float = None, cl_wide : float = 20, # Cl reward parameters
                 delta_reward : bool = False, # Activate the delta reward
                 efficiency_param : float = 1, # Efficiency weight parameter
                 n_boxes : int = 1, boxes : list = None, # Box parameters
                 reynolds : int = 1e6): # Number of boxes in the airfoil
        
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
        - n_boxes: The number of boxes in the airfoil.
        - reynolds: The Reynolds number of the airfoil. If it is -1, it is randomly generated, and if it is None, it is not used.
        """

        
        # Input parameters
        self.max_steps = max_steps
        self.efficiency_th = reward_threshold
        self.cl_reward = cl_reward

        self.cl_reset = cl_reset

        if cl_reward == True and self.cl_reset is not None:
            if self.cl_reset < self._CL_MIN or self.cl_reset > self._CL_MAX:
                raise ValueError(f"cl_reset is out of range. It should be between {self._CL_MIN} and {self._CL_MAX}")
            self.cl_target = self.cl_reset # Cl target is fixed
        else:
            self.cl_target = None # Placeholder for the cl target that will be randomly generated in the reset method


        self.cl_wide = cl_wide
        self.delta_reward = delta_reward
        self.efficiency_param = efficiency_param
        self.scale_actions = scale_actions
        
        

        # Create the airfoil object
        self.state = AirfoilTools() 
        self.n_params = n_params # Number of parameters in one side of the airfoil

        if airfoil_seed is None:
            self.state.kulfan(upper_weights=0.1*np.ones(self.n_params), lower_weights=-0.1*np.ones(self.n_params), leading_edge_weight=0.0) # Initialize the airfoil with the Kulfan parametrization
        elif airfoil_seed is not None and len(airfoil_seed[0]) != self.n_params and len(airfoil_seed[1]) != self.n_params:
            raise ValueError(f"The number of parameters in the airfoil seed should be equal to {self.n_params}")
        
        self.airfoil_seed = airfoil_seed



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

        if self.n_boxes > 0:
            if boxes is not None:
                if len(boxes) != self.n_boxes:
                    raise ValueError(f"The number of boxes should be equal to {self.n_boxes}")
                else:
                    self.boxes = boxes
            else:
                self.boxes = None


        self.reynolds = reynolds

        self.random_reynolds = False # Placeholder for the random reynolds number. If reynolds is -1, it will be True
        self.no_reynolds = False # Placeholder for the case where reynolds is None

        # Check reynolds number
        if self.reynolds is not None:
            if self.reynolds == -1: # It is made because old trained models does not have the reynolds number as an observation.
                self.no_reynolds = True
                self.reynolds = 1e6
            else:
                if self.reynolds < self._RE_MIN or self.reynolds > self._RE_MAX:
                    raise ValueError(f"Reynolds number is out of range. It should be between {self._RE_MIN} and {self._RE_MAX}")
        else:
            self.random_reynolds = True



        # Spaces dict is not used since it means observations are from different types of data. MultiLayerInput 
        # of Stable Baselines 3 is not the most efficient way to handle this. 

        obs_dict = {} # Placeholder for the observation space
        obs_dict["airfoil"] = spaces.Box(low=-5.0, high=5.0, shape=(2*self.n_params + 1,), dtype=np.float32)

        if self.cl_reward == True:
            obs_dict["cl_target"] = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

        if self.n_boxes > 0:
            obs_dict["boxes"] = spaces.Box(low=-5.0, high=5.0, shape=(4*self.n_boxes,), dtype=np.float32)

        if self.no_reynolds == False:
            obs_dict["reynolds"] = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2*self.n_params+1,), dtype=np.float32)




    def _get_info(self) -> dict:
        """
        This method returns additional information about the environment state.
          """
        return {"CL": self.state.get_cl, "efficiency": self.state.get_efficiency, "step": self.step_counter}
    

    def _get_obs(self):
        upper, lower, le = self.state.get_weights()
        observation = {"airfoil": np.array(upper + lower + le, dtype=np.float32),}

        if self.n_boxes > 0:
            boxes_obs = np.zeros(4*self.n_boxes, dtype=np.float32)

            for i in range(self.n_boxes):
                boxes_obs[4*i:4*(i+1)] = self.state.return_boxes()[i]

            observation["boxes"] = boxes_obs

        if self.cl_reward == True:
            observation["cl_target"] = np.array([self.cl_target], dtype=np.float32)

        if self.no_reynolds is False:
            observation["reynolds"] = np.array([self.reynolds / self._RE_MAX], dtype=np.float32) # Normalize the reynolds number between 0 and 1

        return observation




    def reset(self, seed=None, options={}):

        super().reset(seed=seed)
        # Reset the environment state 
        if self.airfoil_seed is not None:
            self.state.kulfan(upper_weights=self.airfoil_seed[0], lower_weights=self.airfoil_seed[1], leading_edge_weight=self.airfoil_seed[2])
        else:
            #raise NotImplementedError("Due to a mistake, it is necessary to implement the airfoil_seed parameter in the reset method. random kulfan needs an airfoil declaration first")
            self.state.random_kulfan2()


        # Box reset
        self.state.boxes = [] # Reset the boxes

        # Box creation
        if self.n_boxes > self._BOX_LIMIT:
            raise ValueError(f"The number of boxes is limited to {self._BOX_LIMIT}")
        
        if self.n_boxes == 0:
            pass

        elif self.n_boxes == 1:
            if self.boxes is None: # If the boxes are not defined, create random boxes
                self.state.get_boxes(BoxRestriction.random_box(y_simmetrical=False, ymin=-0.1, ymax=0.10, widthmax=0.55, heightmax=0.15))
            else: # If the boxes are defined, use them
                self.state.get_boxes(self.boxes[0])

        elif self.n_boxes == 2:
            if self.boxes is None: # If the boxes are not defined, create random boxes
                self.state.get_boxes(BoxRestriction.random_box(y_simmetrical=False, ymin=-0.10, ymax=0.10, widthmax=0.5, heightmax=0.15,
                                                            xmax=0.5))
                self.state.get_boxes(BoxRestriction.random_box(y_simmetrical=False, ymin=-0.10, ymax=0.10, widthmax=0.5, heightmax=0.10,
                                                            xmin=0.5))
            else: # If the boxes are defined, use them
                self.state.get_boxes(self.boxes[0], self.boxes[1])
            
        if self.random_reynolds == True:
            self.reynolds = random.uniform(self._RE_MIN, self._RE_MAX)

        ############################ Temporal solution for the reynolds number ############################

        if 'reynolds' in options:
            self.reynolds = options['reynolds']
            if self.reynolds < self._RE_MIN or self.reynolds > self._RE_MAX:
                raise ValueError(f"Reynolds number is out of range. It should be between {self._RE_MIN} and {self._RE_MAX}")
            print(f"Reynolds number set to {options['reynolds']}. Don't forget to create a setter for the reynolds number in the environment.")

        #############################################################################################

        self.done = False
        self.step_counter = 0

        if self.cl_reward == True and self.cl_reset is None:
            self.cl_target = random.uniform(self._CL_MIN, self._CL_MAX)

        ############################ Temporal solution for the cl target ############################

        if 'cl_target' in options:
            self.cl_target = options['cl_target']
            if self.cl_target < self._CL_MIN or self.cl_target > self._CL_MAX:
                raise ValueError(f"cl_target is out of range. It should be between {self._CL_MIN} and {self._CL_MAX}")

        #############################################################################################

        self.state.analysis(re=self.reynolds) # Analyze the airfoil
        self.last_efficiency = self.state.get_efficiency()

        info = {} # Placeholder for additional information

        #self.state.airfoil_plot() # Plot the airfoil

        return self._get_obs(), info



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
            self.state.analysis(re=self.reynolds) # Analyze the airfoil


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


        #self.state.airfoil_plot() # Plot the airfoil

        return self._get_obs(), self.reward, self.done, False, {"step": self.step_counter, "efficiency": self.state.get_efficiency(),
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