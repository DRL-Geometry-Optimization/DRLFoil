from drlfoil import Train, Optimize
import drlfoil
import drlfoil.utilities

if __name__ == '__main__':
    camion = Train("camion")
    camion.environment_parameters(n_boxes=1)
    camion.training_parameters()
    camion.model_parameters()
    camion.train()