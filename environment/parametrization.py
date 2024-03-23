import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import neuralfoil as nf
import time
import random




class airfoiltools:
    def __init__(self):
        self.airfoil = None # Placeholder for the airfoil object
        self.aerodynamics = None # Placeholder for the aerodynamics of the airfoil 



    def kulfan(self, lower_weights, upper_weights, leading_edge_weight, TE_thickness, name = ""):
        self.airfoil = asb.KulfanAirfoil( # Create the airfoil object with the Kulfan parameterization
        name=name,
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=TE_thickness
        )



    def random_kulfan(self, n_params = 15):
        # Randomize the weights of the airfoil
        np.random.seed(int(time.time())) # Seed the random number generator with the current time
        
        lower_weights = np.random.uniform(low=-1, high=1, size=n_params) # Randomize the lower weights

        upper_weights = np.zeros(n_params) # Initialize the upper weights
        for i in range(len(lower_weights)): # Randomize the upper weights based on the lower weights (to not have intersections)
            upper_weights[i] = random.uniform(lower_weights[i], 1.1)
            
        leading_edge_weight = np.random.random()
        TE_thickness = 0 # Thickness of the trailing edge 
        
        # Create the airfoil
        self.kulfan(lower_weights, upper_weights, leading_edge_weight, TE_thickness)



    def analysis(self, angle = 0, re = 1e6, model = "xlarge"): # Analyze the airfoil and save into the aerodynamics attribute (dictionary)
        self.aerodynamics = nf.get_aero_from_kulfan_parameters(
            self.airfoil.kulfan_parameters, 
            angle, re, 
            model)



    def airfoil_plot(self):
        fig, ax = plt.subplots(figsize=(6, 2))
        self.airfoil.draw()


