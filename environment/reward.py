from parametrization import airfoiltools

def reward(efficiency, last_efficiency = None, cl = None, cd = None, delta_reward = True, cl_target = False, cd_target = False):

    if delta_reward == True:
        if cl_target == False & cd_target == False:
            try:
                return efficiency - last_efficiency
            except:
                raise TypeError("Reward could not be calculated. The value of Last Efficiency was not introduced and Delta Reward is activated")
        else:
            print("Warning: cl_target or cd_target = True was not defined yet as a reward function. Returning delta efficiency as reward")
            return efficiency - last_efficiency
    else:
        if cl_target == False & cd_target == False:
            return efficiency
        else:
            print("Warning: cl_target or cd_target = True was not defined yet as a reward function. Returning efficiency as reward")
            return efficiency
    


    
