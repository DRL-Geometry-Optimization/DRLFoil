import numpy as np

def reward(efficiency : float, efficiency_param : float = 0.5, 
           cl_reward : bool = False, cl : float = None, cl_target : float = None, cl_wide : float = 8, 
           delta_reward : bool = False, last_efficiency : float = None):
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
            try:
                delta_Cl = cl - cl_target
                return (efficiency - last_efficiency)*efficiency_param*np.exp(-cl_wide*(delta_Cl)**2)
            except:
                raise TypeError("Reward could not be calculated. The value of Last Efficiency was not introduced and Delta Reward is activated")
        

    # Delta Reward False
    else:
        # Cl target False
        if cl_reward == False:
            return efficiency_param*efficiency
        # Cl target True
        else:
            delta_Cl = cl - cl_target
            return efficiency_param*efficiency*np.exp(-cl_wide*(delta_Cl)**2)
    


    
