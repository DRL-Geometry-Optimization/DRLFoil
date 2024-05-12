from ..airfoil_env.parametrization import AirfoilTools
import numpy as np
import matplotlib.pyplot as plt

def AeroAnalysis(airfoil : AirfoilTools, reynolds : int = 1e6, plot : bool = False):
    alpha_list = []
    cl_list = []
    cd_list = []
    efficiency_list = []

    for alpha in range(-15, 30):
        airfoil.analysis(angle=alpha, re=reynolds)
        alpha_list.append(alpha)
        cl_list.append(airfoil.get_cl())
        cd_list.append(airfoil.get_cd())
        efficiency_list.append(airfoil.get_efficiency())

    if plot:
        plt.plot(alpha_list, cl_list, label='Cl')
        plt.legend()
        plt.show()
        plt.plot(alpha_list, cd_list, label='Cd')
        plt.legend()
        plt.show()
        plt.plot(alpha_list, efficiency_list, label='Efficiency')
        plt.legend()
        plt.show()

    return {'alpha' : alpha_list, 'cl' : cl_list, 'cd' : cd_list, 'efficiency' : efficiency_list}


    """
    for alpha in range(-10, 25):
        airfoil.analysis(angle=alpha, re=reynolds)
        cl = airfoil.get_cl()
        if cl > maxcl:
            maxcl = cl
            maxcl_alpha = alpha
            maxcl_eff = airfoil.get_efficiency()"""
        
    print("Cl list: ", cl_list)
    print("Max Cl: ", max(cl_list))