from gymnasium.envs.registration import register

register(
    id='AirfoilEnv-v0',
    entry_point='airfoil_env.gym_environment:AirfoilEnv',
)


from .gym_environment import AirfoilEnv
from .parametrization import AirfoilTools
from .restriction import BoxRestriction, plot_boxes
from .reward import reward