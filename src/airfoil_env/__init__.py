from gymnasium.envs.registration import register

register(
    id='AirfoilEnv-v0',
    entry_point='airfoil_env.gym_environment:AirfoilEnv',
)