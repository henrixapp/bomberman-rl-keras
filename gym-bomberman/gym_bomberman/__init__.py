from gym.envs.registration import register

register(
    id='bomberman-v0',
    entry_point='gym_bomberman.envs:BombermanEnv',
)