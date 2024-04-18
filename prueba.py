from environment.parametrization import airfoiltools
from environment.restriction import BoxRestriction

for _ in range(5):
    prueba = airfoiltools()
    prueba.kulfan(upper_weights=[0.5, 0.5], lower_weights=[-0.5, -0.5], leading_edge_weight=0)
    prueba.get_boxes(BoxRestriction.random_box(xmax=0.5), BoxRestriction.random_box(xmin=0.5))
    prueba.airfoil_plot()