import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import time
import random




class airfoil_param:
    def __init__(self, n_params):
        self.n_params = n_params
        self.airfoil = None

    def kulfan(self, lower_weights, upper_weights, leading_edge_weight, TE_thickness):
        self.airfoil = asb.KulfanAirfoil(
        name="My Airfoil",
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=TE_thickness
        )

    def random_kulfan(self, n_params = 50):
        np.random.seed(int(time.time()))
        
        lower_weights = np.random.uniform(low=-1, high=1, size=n_params) # Randomize the lower weights

        upper_weights = np.zeros(n_params) # Initialize the upper weights
        for i in range(len(lower_weights)): # Randomize the upper weights based on the lower weights (to not have intersections)
            upper_weights[i] = random.uniform(lower_weights[i], 1.1)
            

        leading_edge_weight = np.random.random()
        TE_thickness = 0 # Thickness of the trailing edge 
        self.airfoil = asb.KulfanAirfoil(
        name="My Airfoil",
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=TE_thickness
        )




#### TEST ####



pedro = airfoil_param(4)

pedro.kulfan(
    lower_weights = -0.6 * np.ones(8),
    upper_weights = 0.2 * np.ones(8),
    leading_edge_weight = 0.15,
    TE_thickness = 0
)
"""
fig, ax = plt.subplots(figsize=(6, 2))
pedro.airfoil.draw()
"""
pedro.random_kulfan()

fig, ax = plt.subplots(figsize=(6, 2))
pedro.airfoil.draw()

