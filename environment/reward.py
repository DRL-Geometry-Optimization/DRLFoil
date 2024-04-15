import numpy as np

def reward(efficiency, efficiency_param=1, 
           cl_reward = False, cl = None, cl_target = None, cl_maxreward = 40, cl_wide = 10, 
           delta_reward = False, last_efficiency = None):
    # Delta Reward True
    if delta_reward == True:
        # Cl target False
        if cl_reward == False:
            try:
                return (efficiency - last_efficiency)*efficiency_param
            except:
                raise TypeError("Reward could not be calculated. The value of Last Efficiency was not introduced and Delta Reward is activated")
        # Cl target True
        else:
            print("Warning: cl_target or cd_target = True was not defined yet as a reward function. Returning delta efficiency as reward")
            return (efficiency - last_efficiency)*efficiency_param
        

    # Delta Reward False
    else:
        # Cl target False
        if cl_reward == False:
            return efficiency_param*efficiency
        # Cl target True
        # NOTE: IF CL_TARGET IS ACTIVATED, THE NEURAL NETWORK SHOULD HAVE THE CL TARGET AS INPUT
        else:
            delta_Cl = cl - cl_target
            return efficiency_param*efficiency + cl_maxreward*np.exp(-cl_wide*(delta_Cl)**2)
    


    
