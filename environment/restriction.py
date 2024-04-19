import numpy as np
import matplotlib.pyplot as plt

class BoxRestriction:
    """
    Box restriction class. It is used to create a space constraint in the airfoil that the optimization algorithm must respect.
    The box is defined by its center position (posx, posy), width and height.
    """

    _XMIN = 0.1
    _XMAX = 0.8
    _YMIN = -0.3
    _YMAX = 0.3

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

        # Check if the box is valid        
        if not isinstance(posx, (int, float)) or not isinstance(posy, (int, float)):
            raise ValueError("Position must be a number")
        
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            raise ValueError("Width and height must be numbers")
        
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")      

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

        if self.coordinates()[0][0] < self._XMIN or self.coordinates()[1][0] > self._XMAX:
            raise ValueError(f"Box is out of the environment. It must be between {self._XMIN} and {self._XMAX} in x axis")
        
        if self.coordinates()[0][1] > self._YMAX or self.coordinates()[2][1] < self._YMIN:
            raise ValueError(f"Box is out of the environment. It must be between {self._YMIN} and {self._YMAX} in y axis")  

    def __str__(self) -> str:
        """
        String representation of the BoxRestriction class.
        """
        return f"BoxRestriction(posx = {self.posx}, posy = {self.posy}, width = {self.width}, height = {self.height})"
    


    def __getitem__(self, key : int) -> float:
        """
        Items:
        0 : posx
        1 : posy
        2 : width
        3 : height
        """
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
    
    def box_params(self):
        pass

    def coordinates(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        Returns the coordinates of the box in the form of a tuple of (x,y).
        0: up_left
        1: up_right
        2: down_left
        3: down_right
        """
        up_left = (self.posx - self.width/2, self.posy + self.height/2)
        up_right = (self.posx + self.width/2, self.posy + self.height/2)
        down_left = (self.posx - self.width/2, self.posy - self.height/2)
        down_right = (self.posx + self.width/2, self.posy - self.height/2)

        return up_left, up_right, down_left, down_right
    


    def plot(self, show : bool = False) -> None:
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
        """
        Checks if a point is inside the box.

        Args:
        x : float
            x coordinate of the point.
        y : float
            y coordinate of the point.
        """
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def is_bad(self, x : float, y : float, face: str) -> bool:
        """
        Checks if a point of the airfoil is not correctly possitioned with respect to the restriction box

        Args:
        x : float
            x coordinate of the point.
        y : float
            y coordinate of the point.
        """
        if face == "up":
            return self.x1 <= x <= self.x2 and y <= self.y2
        
        if face == "low":
            return self.x1 <= x <= self.x2 and self.y1 <= y
    


    @classmethod
    def random_box(cls, xmin : float = None, xmax : float = None, 
                   ymin : float = None, ymax : float = None,
                   widthmax : float = None, heightmax : float = None,
                   y_simmetrical : bool = True) -> 'BoxRestriction':
        """
        Creates a random box.

        Args:
        xmin : float
            Minimum x position of the box. Default is 0.1.
        xmax : float
            Maximum x position of the box. Default is 0.9.
        ymin : float
            Minimum y position of the box. Default is -0.2.
        ymax : float
            Maximum y position of the box. Default is 0.2.
        widthmax : float
            Maximum width of the box. Default is xmax - xmin.
        heightmax : float
            Maximum height of the box. Default is ymax - ymin.
        y_simmetrical : bool
            If True, the box will be in the middle of the y axis. Default is True.
        """

        # Set the mins and maxs with defaults values if they are None
        if xmin is None:
            xmin = cls._XMIN
        if xmax is None:
            xmax = cls._XMAX
        if ymin is None:
            ymin = cls._YMIN
        if ymax is None:
            ymax = cls._YMAX

        # Set the width and height to not exceed the limits, which can be set by the user with widthmax and heightmax or the default values
        if widthmax is None:
            widthmax = xmax - xmin
        else:
            if widthmax < 0:
                raise ValueError("Widthmax must be positive")
            widthmax = min(widthmax, xmax - xmin) # It takes the most restrictive value

        # heightmax is the same as widthmax
        if heightmax is None:
            heightmax = ymax - ymin
        else:
            if heightmax < 0:
                raise ValueError("Heightmax must be positive")
            heightmax = min(heightmax, ymax - ymin)

        width = np.random.uniform(0.1, widthmax)
        height = np.random.uniform(0.1, heightmax)

        # After setting the width and height, it sets the position of the box according to the limits
        posx = np.random.uniform(xmin + width/2, xmax - width/2)

        if y_simmetrical:
            posy = 0
        else:
            posy = np.random.uniform(ymin + height/2, ymax - height/2)

        return cls(posx, posy, width, height)
    


def plot_boxes(*boxes : BoxRestriction) -> None:
    plt.figure(figsize=(10, 7))
    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    for box in boxes:
        if not isinstance(box, BoxRestriction):
            raise TypeError("All arguments must be BoxRestriction objects")
        box.plot()
    plt.show()





if __name__ == "__main__":
    # Test of the BoxRestriction class
    pedro = BoxRestriction(0.6, 0.2, 0.3, 0.1)
    print(pedro)
    print(pedro.coordinates())
    print(pedro[0])
    print(pedro.is_inside(0.75, 0.2))
    pedro.plot(show=True)

    #plot_boxes(pedro)

