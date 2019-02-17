from gym.envs.registration import register
from gym_bomberman.envs import BombermanEnv
register(
    id='bomberman-v0',
    entry_point='gym_bomberman.envs:BombermanEnv'
)