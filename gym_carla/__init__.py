from gym.envs.registration import register

register(
    id='CarlaEnv-v0',
    entry_point='gym_carla.envs:CarlaEnv',
)
