from environment.parametrization import airfoiltools
import numpy as np

prueba = airfoiltools()

upper_weights = np.array([0.12, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

lower_weights = -1*np.array([0, -0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

prueba.kulfan(upper_weights=upper_weights, lower_weights=lower_weights, leading_edge_weight=0.)

print(prueba.check_airfoil())

up, low = prueba.get_coordinates()
print(up[99])
print(low[1])

prueba.airfoil_plot()