from gym.envs.registration import register
from gym_bomberman.envs import BombermanEnv
from gym_bomberman.envs import CoinmanEnv
register(
    id='bomberman-v0',
    entry_point='gym_bomberman.envs:BombermanEnv'
)
register(
    id='coinman-v0',
    entry_point='gym_bomberman.envs:CoinmanEnv'
)