import numpy as np
import matplotlib.pyplot as plt

class BoxRestriction:
    """
    Box restriction class. It is used to create a space constraint in the airfoil that the optimization algorithm must respect.
    The box is defined by its center position (posx, posy), width and height.
    """
    def __init__(self, posx : float, posy : float, width : float, height : float) -> None:
        """
        Constructor of the BoxRestriction class.

        Parameters:
        posx : float
            Center x position of the box. It must be between 0.1 and 0.9.
        posy : float
            Center y position of the box. It must be between -0.2 and 0.2.
        width : float
            Width of the box. It must be positive.
        height : float
            Height of the box. It must be positive.
        """

        # Characteristics of the box
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height

        # Linear coordinates of the box
        self.x1 = posx - width/2
        self.x2 = posx + width/2
        self.y1 = posy - height/2
        self.y2 = posy + height/2

        # Check if the box is valid
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        
        if not isinstance(posx, (int, float)) or not isinstance(posy, (int, float)):
            raise ValueError("Position must be a number")
        
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            raise ValueError("Width and height must be numbers")
        
        if posx > 0.9 or posx < 0.1:
            raise ValueError("Position x must be between 0.1 and 0.9")
        
        if posy > 0.2 or posy < -0.2:
            raise ValueError("Position y must be between -0.2 and 0.2")
        
        if self.coordinates()[0][0] < 0 or self.coordinates()[1][0] > 1:
            raise ValueError("Box is out of the environment. It must be between 0 and 1 in x axis")
        
        if self.coordinates()[0][1] > 0.4 or self.coordinates()[2][1] < -0.4:
            raise ValueError("Box is out of the environment. It must be between -0.4 and 0.4 in y axis")        



    def __str__(self) -> str:
        """
        String representation of the BoxRestriction class.
        """
        return f"BoxRestriction(posx = {self.posx}, posy = {self.posy}, width = {self.width}, height = {self.height})"
    


    def __getitem__(self, key : int) -> float:
        if key == 0:
            return self.posx
        elif key == 1:
            return self.posy
        elif key == 2:
            return self.width
        elif key == 3:
            return self.height
        else:
            raise IndexError("Index out of range. It must be between 0 and 3")
    


    def coordinates(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        Returns the coordinates of the box in the form of a tuple of (x,y).
        """
        up_left = (self.posx - self.width/2, self.posy + self.height/2)
        up_right = (self.posx + self.width/2, self.posy + self.height/2)
        down_left = (self.posx - self.width/2, self.posy - self.height/2)
        down_right = (self.posx + self.width/2, self.posy - self.height/2)

        return up_left, up_right, down_left, down_right
    


    def plot(self, show : bool = True) -> None:
        """
        Plots the box in the environment. If show is True, it will show the plot.

        Parameters:
        show : bool
            If True, it will show the plot. Default is True.
        """
        up_left, up_right, down_left, down_right = self.coordinates()
        x = [up_left[0], up_right[0], down_right[0], down_left[0], up_left[0]]
        y = [up_left[1], up_right[1], down_right[1], down_left[1], up_left[1]]
        plt.plot(x, y)

        if show:
            plt.show()



    def is_inside(self, x : float, y : float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    




def plot_boxes(*boxes):
    for box in boxes:
        box.plot()
    plt.show()


pedro = BoxRestriction(0.5, 0, 0.2, 0.2)
pedro2 = BoxRestriction(0.3, 0, 0.2, 0.2)
print(pedro, pedro2)

print(pedro[0])