import numpy as np
from environment.parametrization import airfoiltools






pedro = airfoiltools()
"""
pedro.kulfan(
    lower_weights = -0.2 * np.ones(8),
    upper_weights = 0.1 * np.ones(8),
    leading_edge_weight = 0.1,
    TE_thickness = 0
)
"""

pedro.random_kulfan(20)

pedro.airfoil_plot()

pedro.analysis(0, 1e6, "xlarge")
print(pedro.aerodynamics)