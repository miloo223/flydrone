from gym.envs.registration import register

register(
    id='fly_drone-v0',
    entry_point='fly_drone.envs:Fly_drone',
)