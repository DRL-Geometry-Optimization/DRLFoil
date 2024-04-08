from gymnasium.envs.registration import register

register(
    id='Airfoil-v0',
    entry_point='gymnasium.envs:AirfoilEnv',
    max_episode_steps=300,
    reward_threshold=25.0,
)