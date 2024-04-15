from gymnasium.envs.registration import register

register(
    id='AirfoilEnv-v0',
    entry_point='environment.gym_environment:AirfoilEnv',
)