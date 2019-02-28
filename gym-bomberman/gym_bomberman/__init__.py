from gym.envs.registration import register
from gym_bomberman.envs import BombermanEnv
from gym_bomberman.envs import BombermanDieHardEnv
from gym_bomberman.envs import CoinmanEnv
from gym_bomberman.envs import Coinman2Env
register(
    id='bomberman-v0',
    entry_point='gym_bomberman.envs:BombermanEnv'
)
register(
    id='bombermandiehard-v0',
    entry_point='gym_bomberman.envs:BombermanDieHardEnv'
)
register(
    id='coinman-v0',
    entry_point='gym_bomberman.envs:CoinmanEnv'
)
register(
    id='coinman2-v0',
    entry_point='gym_bomberman.envs:Coinman2Env'
)