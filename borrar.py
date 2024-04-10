from environment.parametrization import airfoiltools
import matplotlib.pyplot as plt


testing = airfoiltools()

upparameters = [0.1, 
                0.1, 
                0.1, 
                0.1, 
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,]

downparameters = [0.2, 
                -0.1, 
                -0.1, 
                -0.1, 
                +0.6,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,]

le_weight = 0.0

#testing.kulfan(upparameters, downparameters, le_weight)

#testing.airfoil_plot()
print(testing.check_airfoil())

#testing.airfoil_plot()
"""
a, b = testing.get_coordinates()
lenn = len(a)
if lenn == len(b):
    print("True")
print(a[50][1], b[lenn-51][1])
#print("xD")
#print(b)


upper = testing.airfoil.upper_coordinates()
lower = testing.airfoil.lower_coordinates()

x, y = zip(*upper)

plt.plot(x[10], y[10], label = "Upper side")

x2, y2 = zip(*lower)

plt.plot(x2[10], y2[10], label = "Lower side")

plt.show()
"""