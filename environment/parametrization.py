import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import random




class airfoil_param:
    def __init__(self, n_params):
        self.n_params = n_params
        self.airfoil = None

    def kulfan(self, lower_weights, upper_weights, leading_edge_weight, TE_thickness):
        self.airfoil = asb.KulfanAirfoil(
        name="My Airfoil",
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=TE_thickness
        )

    def random_kulfan(self):
        lower_weights = np.random.random()
        upper_weights = np.random.random()
        leading_edge_weight = np.random.random()
        TE_thickness = np.random.random()
        self.airfoil = asb.KulfanAirfoil(
        name="My Airfoil",
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=TE_thickness
        )




#### TEST ####



pedro = airfoil_param(4)

pedro.kulfan(
    lower_weights = -0.2 * np.ones(8),
    upper_weights = 0.3 * np.ones(8),
    leading_edge_weight = 0.2,
    TE_thickness = 0.005
)

fig, ax = plt.subplots(figsize=(6, 2))
pedro.airfoil.draw()

pedro.random_kulfan()

fig, ax = plt.subplots(figsize=(6, 2))
pedro.airfoil.draw()