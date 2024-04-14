import environment
from environment.parametrization import airfoiltools

class ModelManagement():
    def __init__(self, model, env) -> None:
        self.model = model
        self.env = env
        self.airfoil = None

    def get_airfoil(self, airfoil):
        self.airfoil = airfoiltools()
        self.airfoil.kulfan(airfoil[0], airfoil[1], airfoil[2], airfoil[3], airfoil[4])

        def predict(self):
            pass
    
