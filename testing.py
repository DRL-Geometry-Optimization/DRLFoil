import numpy as np
from environment.parametrization import airfoiltools






pedro = airfoiltools()

pedro.kulfan(
    lower_weights = -0.2 * np.ones(15),
    upper_weights = 0.1 * np.ones(9),
    leading_edge_weight = 0.1,
    TE_thickness = 0
)


pedro.airfoil_plot()

pedro.modify_airfoil("up", 20, 0.3)

pedro.airfoil_plot()