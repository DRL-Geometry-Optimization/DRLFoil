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


    def random_kulfan(self, n_params = 15, variation = 0.1, thickness = 1.1):
        # Randomize the weights of the airfoil
        np.random.seed(int(time.time())) # Seed the random number generator with the current time

        leading_edge_weight = random.uniform(-variation*3, variation*3) # Randomize the leading edge weight
        TE_thickness = 0 # Thickness of the trailing edge
        
        lower_weights = np.zeros(n_params) # Initialize the lower weights
        
        for i in range(len(lower_weights)): # Randomize the lower weights
            if i == 0: # Skip the first weight
                lower_weights[i] = random.uniform(-leading_edge_weight-variation, -leading_edge_weight+variation) # Randomize the first weight
            else:
                lower_weights[i] = random.uniform(lower_weights[i-1] - variation, lower_weights[i-1] + variation)

        upper_weights = np.zeros(n_params) # Initialize the upper weights
        for i in range(len(lower_weights)): # Randomize the upper weights based on the lower weights (to not have intersections)
            if i == 0:
                upper_weights[i] = random.uniform(lower_weights[i], lower_weights[i]+variation)
            else:
                upper_weights[i] = random.uniform(lower_weights[i], upper_weights[i-1] + variation)

        lower_weights = lower_weights * thickness # Scale the lower weights
        upper_weights = upper_weights * thickness

        # Create the airfoil
        self.kulfan(lower_weights, upper_weights, leading_edge_weight, TE_thickness)



    def random_kulfan2(self, n_params = 15, variation = 0.1, extra_weight = 0.7, intra_weight = 0.3):
        # Randomize the weights of the airfoil
        np.random.seed(int(time.time()))

        leading_edge_weight = random.uniform(-variation*3, variation*3) # Randomize the leading edge weight
        TE_thickness = 0 # Thickness of the trailing edge

        lower_weights = np.zeros(n_params) # Initialize the lower weights

        for i in range(len(lower_weights)): # Randomize the lower weights
            lower_weights[i] = random.uniform(-intra_weight-variation, -intra_weight+variation)
        
        upper_weights = np.zeros(n_params) # Initialize the upper weights
        for i in range(len(lower_weights)): # Randomize the upper weights based on the lower weights (to not have intersections)
            if extra_weight-variation < lower_weights[i]:
                upper_weights[i] = random.uniform(lower_weights[i], extra_weight+variation)
            else:
                upper_weights[i] = random.uniform(extra_weight-variation, extra_weight+variation)

        print(lower_weights)
        print(upper_weights)

        # Create the airfoil
        self.kulfan(lower_weights, upper_weights, leading_edge_weight, TE_thickness)
            
         
        



    """
    def random_kulfan(self, n_params = 15, variation = 0.1):
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
    """


    def analysis(self, angle = 0, re = 1e6, model = "xlarge"): # Analyze the airfoil and save into the aerodynamics attribute (dictionary)
        self.aerodynamics = nf.get_aero_from_kulfan_parameters(
            self.airfoil.kulfan_parameters, 
            angle, re, 
            model)



    def airfoil_plot(self):
        fig, ax = plt.subplots(figsize=(6, 2))
        self.airfoil.draw()


