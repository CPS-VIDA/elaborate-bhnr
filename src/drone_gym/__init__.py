from gym.envs.registration import register


register(
    id='SimpleQuadrotorPositionControlEnv-v0',
    entry_point='drone_gym.envs:SimpleQuadrotorPositionControlEnv',
    max_episode_steps=2000,
)

register(
    id='VREPQuadrotorPositionControlEnv-v0',
    entry_point='drone_gym.envs:VREPQuadrotorPositionControlEnv',
    max_episode_steps=1000,
)