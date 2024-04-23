import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import neuralfoil as nf
import time
import random
from .restriction import BoxRestriction




class airfoiltools:
    """
    This class provides tools for working with one airfoil. It can create an airfoil with the Kulfan parameterization
    and analyze. It also provides methods for modifying the airfoil and getting the weights of the airfoil.
    """

    def __init__(self):
        """
        Initializes the Parametrization class.
        """
        self.airfoil = None # Placeholder for the airfoil object
        self.aerodynamics = None # Placeholder for the aerodynamics of the airfoil 
        self.numparams = None # Placeholder for the number of parameters on each face
        self.boxes = [] # Placeholder for the box restriction


    @property
    def upper_weights(self) -> list:
        """
        Returns the upper weights of the airfoil. 
        """
        return self.airfoil.upper_weights
    
    @property
    def lower_weights(self) -> list: 
        """
        Returns the lower weights of the airfoil.
        """
        return self.airfoil.lower_weights


    def kulfan(self, upper_weights: list, lower_weights: list, leading_edge_weight: float, TE_thickness: float = 0, 
               name: str = "") -> None: 
        """
        This method creates an airfoil object with the Kulfan parameterization. It will be saved at self.airfoil.

        Args:
            upper_weights: A list of floats representing the weights of the upper side of the airfoil.
            lower_weights: A list of floats representing the weights of the lower side of the airfoil.
            leading_edge_weight: A float representing the weight of the leading edge of the airfoil.
            TE_thickness: A float representing the thickness of the trailing edge of the airfoil.
            name: A string representing the name of the airfoil.
        """


        if len(lower_weights) != len(upper_weights):
            raise ValueError("The number of weights in the upper and lower side of the airfoil must be the same")

        self.numparams = len(upper_weights) # Number of parameters in one side of the airfoil

        self.airfoil = asb.KulfanAirfoil( # Create the airfoil object with the Kulfan parameterization
        name=name,
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=TE_thickness
        )




    def random_kulfan(self, variation: float = 0.1, thickness: float = 1.1) -> None:
        """
        Randomizes the weights of the airfoil with the Kulfan parameterization. 
        The weights are randomized based on the thickness of the airfoil.

        Args:
            variation: A float representing the variation of the weights.
            thickness: A float representing the thickness multiplier for the airfoil.
        """
        
        np.random.seed(int(time.time())) # Seed the random number generator with the current time

        leading_edge_weight = random.uniform(-variation*3, variation*3) # Randomize the leading edge weight
        TE_thickness = 0 # Thickness of the trailing edge
        
        lower_weights = np.zeros(self.numparams) # Initialize the lower weights
        
        for i in range(len(lower_weights)): # Randomize the lower weights
            if i == 0: # Skip the first weight
                lower_weights[i] = random.uniform(-leading_edge_weight-variation, -leading_edge_weight+variation) # Randomize the first weight
            else:
                lower_weights[i] = random.uniform(lower_weights[i-1] - variation, lower_weights[i-1] + variation)

        upper_weights = np.zeros(self.numparams) # Initialize the upper weights
        for i in range(len(lower_weights)): # Randomize the upper weights based on the lower weights (to not have intersections)
            if i == 0:
                upper_weights[i] = random.uniform(lower_weights[i], lower_weights[i]+variation)
            else:
                upper_weights[i] = random.uniform(lower_weights[i], upper_weights[i-1] + variation)

        lower_weights = lower_weights * thickness # Scale the lower weights
        upper_weights = upper_weights * thickness

        # Create the airfoil
        self.kulfan(lower_weights, upper_weights, leading_edge_weight, TE_thickness)



    # Randomize the airfoil with a different method. This methods parts from the upper and lower weights and randomizes them separately
    # Randomize the weights of the airfoil   
    def random_kulfan2(self, variation: float = 0.5, extra_weight: float = 0.5, intra_weight: float = 0.2) -> None: 
        """
        Another method to randomize the weights of the airfoil with the Kulfan parameterization. 
        This method randomizes the weights separately.

        Args:
            variation: A float representing the variation of the weights.
            extra_weight: A float representing the extra weight of the airfoil.
            intra_weight: A float representing the intra weight of the airfoil.
        """

        np.random.seed(int(time.time()))

        leading_edge_weight = random.uniform(-variation*3, variation*3) # Randomize the leading edge weight
        TE_thickness = 0 # Thickness of the trailing edge

        lower_weights = np.zeros(self.numparams) 

        for i in range(len(lower_weights)):
            lower_weights[i] = random.uniform(-intra_weight-variation, -intra_weight+variation) # Randomize the first weight
        

        upper_weights = np.zeros(self.numparams) 

        for i in range(len(lower_weights)): 
            if extra_weight-variation < lower_weights[i]: # if lower weight can be higher than the extra weight
                upper_weights[i] = random.uniform(lower_weights[i], extra_weight+variation)
            else:
                upper_weights[i] = random.uniform(extra_weight-variation, extra_weight+variation)

        # Create the airfoil
        self.kulfan(lower_weights=lower_weights, upper_weights=upper_weights, leading_edge_weight=leading_edge_weight, TE_thickness=TE_thickness)



    def modify_airfoil(self, action: np.ndarray, TE_thickness: float = 0, name: str = "") -> None:
        """
        Modify the airfoil by changing the weights. The weights are added to the existing weights of the airfoil.

        Args:
            action: A list of floats representing the weights of the airfoil.
            TE_thickness: A float representing the thickness of the trailing edge of the airfoil.
            name: A string representing the name of the airfoil.
        """

        # Split the action into the upper weights, lower weights, and leading edge weight 
        # NOTE: this is a so bad way to do it, but it works FOR NOW. It should be changed in the future
        act = [action[:self.numparams], action[self.numparams:2*self.numparams], action[-1]]

        self.airfoil.upper_weights = self.airfoil.upper_weights + act[0]
        self.airfoil.lower_weights = self.airfoil.lower_weights + act[1]
        self.airfoil.leading_edge_weight = self.airfoil.leading_edge_weight + act[2]
        self.airfoil.TE_thickness = self.airfoil.TE_thickness + TE_thickness
        self.airfoil.name = name # Change the name of the airfoil if needed



    def modify_airfoil_unit(self, face: str, index: int, variation: float) -> None:  
        """
        Modify the airfoil by changing the weight of a parameter.

        Args:
            face: A string representing the face of the airfoil ("up" or "down").
            index: An integer representing the index of the parameter.
            variation: A float representing the variation of the parameter.
        """

        try:
            if face == "up":
                self.airfoil.upper_weights[index] = self.airfoil.upper_weights[index] + variation
            elif face == "down":
                self.airfoil.lower_weights[index] = self.airfoil.lower_weights[index] + variation
            else:
                raise ValueError("Invalid face. Please, use 'up' or 'down'") # Raise an error if the face is not up or down
        except IndexError: # except the index error if the index is out of bounds
            print(f"Index out of bounds. Probably the airfoil has less weights than {index}")
        except Exception as e: # except any other error
            print(f"Error: {e}")
            


    def analysis(self, angle: float = 0, re: int = 1e6, model: str = "xlarge") -> None:
        """
        Analyze the airfoil and save the aerodynamics into the object self.aerodynamics as a dictionary.

        Args:
            angle: A float representing the angle of attack of the airfoil.
            re: An integer representing the Reynolds number of the airfoil.
            model: A string representing the model of the airfoil.
        """
        self.aerodynamics = nf.get_aero_from_kulfan_parameters(
            self.airfoil.kulfan_parameters, 
            angle, re,
            model)



    def get_cl(self) -> float:  
        """
        Returns the lift coefficient of the current airfoil.
        """
        if self.aerodynamics is None:
            # Raise an error shoud be changed if you do not want to stop the program. The other option is to return None
            raise ValueError("Please, analyze the airfoil first") 
        return self.aerodynamics["CL"][0]
        


    def get_cd(self) -> float:  
        """
        Returns the drag coefficient of the current airfoil.
        """
        if self.aerodynamics is None:
            raise ValueError("Please, analyze the airfoil first")
        return self.aerodynamics["CD"][0]


    def get_efficiency(self) -> float:
        """
        Returns the efficiency of the airfoil.
        """
        try:
            cl = self.get_cl()
            cd = self.get_cd()
            return cl/cd
        except ValueError as e:
            raise ValueError(f"An unexpected error occurred while obtaining efficiency: {e}")
        

    def airfoil_plot(self, box: bool = True) -> None:
        """
        Plot the airfoil using the Matplotlib library.
        """
        fig, ax = plt.subplots(figsize=(6, 2))
        for box in self.boxes:
            box.plot()
        self.airfoil.draw()


    def get_weights(self) -> tuple:
        """
        Returns the weights of the airfoil as a tuple of lists with the form: [upper_weights, lower_weights, leading_edge_weight]
        """
        return self.upper_weights.tolist(), self.lower_weights.tolist(), [self.airfoil.leading_edge_weight]


    def get_coordinates(self) -> tuple: # Get the coordinates of the airfoil
        return self.airfoil.upper_coordinates(), self.airfoil.lower_coordinates()
    
 
    def check_airfoil(self) -> bool:
        """
        Check if the airfoil is valid. The upper side of the airfoil must be above the lower side.
        Returns True if the airfoil is valid, otherwise returns False.
        """
        if self.airfoil is None:
            raise ValueError("Please, create an airfoil first")
        else:
            up, low = self.get_coordinates()
            for i in range(len(up)):
                # The loop goes through all the points of the airfoil and checks if the upper side is above the lower side
                #if up[i][0] != low[-i-1][0]:
                #    raise ValueError("Something went wrong. The airfoil coordinates are not aligned. Please, check the airfoil coordinates.")
                if up[i][1] < low[-i-1][1]:
                    return False
                
                if i > 3 and i < 10: # Check if the airfoil is too thin in the middle leading edge
                    if up[-i-1][1] - low[i][1] < 0.01:
                        return False
                    
                if i == 1 and low[i][1] > 0.008: # Check if the airfoil is like the nose of a bird
                    return False
                
                for box in self.boxes:
                    if box.is_bad(x = up[-i-1][0], y = up[-i-1][1], face = "up") or box.is_bad(x = low[i][0], y = low[i][1], face = "low"):
                        return False

            # If the airfoil is valid, return True    
            return True
        
    def get_boxes(self, *boxes) -> None:
        for box in boxes:
            if not isinstance(box, BoxRestriction):
                raise TypeError("The box must be an instance of the BoxRestriction class")
            self.boxes.append(box)

    def return_boxes(self) -> list:
        return_boxes = []
        for box in self.boxes:
            return_boxes.append([box.posx, box.posy, box.width, box.height])
        return return_boxes
    







if __name__ == "__main__": #This will only run if the script is run directly, not imported
    prueba = airfoiltools()
    prueba.get_boxes(BoxRestriction(0, 0, 0.3, 0.2))
    print(prueba.boxes)