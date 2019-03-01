# Gym_bomberman

Provides a package for use with certain algorithms

##  Installation

###  Enviroment

Create an new conda-env called bomberman and python 3.6 or 3.7:

```sh
conda create -n bomberman python=3
conda activate bomberman
```

Install packages:
 
 1. For A3C--implementation
```sh
conda install matplotlib
pip install tensorflow
# Finally install our own package gym_bomberman (you have to be in gym-bomberman)
pip install -e .
```
2. For DQN--Agent and keras-rl add:
```sh
pip install pillow keras-rl
```

## Enviroments

### General Idea

We provide an gym env for training bomberman. A discrete gym env roughly has:
- `action_space`  which action can we feed in to our env?
- `observation_space` How does the observation look like, which values are possible?
- `step(self,action)` Simulates one step, given the action, in our case an uint8 from 0 to 5. Returns whether the game exited, the reward for those actions and the observation
- `render(self,mode=human)` Renders the current state (in our case into the terminal)

#### Our action_space

```python
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
BOMB = 4
WAIT = 5
```

#### our observationspace:

| Output | Rendered | meaning |
| -------- | ----------- | ---------- |
|     -3     |       💥     | Explosion |
|    -2      |      💣      |  Bomb    |
|    -1      |      ❌      |  Wall (not destroyable) |
|     0      |      👣      |  Free  |
|     1      |     ❎       | Crate (destroyable) |
|     2      |     🏆       | Coin  |
|     3      |     😎      |  Player |

### List of aviable envs
All enviroments are currently single player

1. `bomberman-v0`
    - Implementation of the game bomberman (17x17)  which was provided by the lecturer.
    - Perspective will be limited to 16x16 field in DQN
    - not yet used with a3c
    - should terminate later after 400 steps
2. `coinman-v0`
    - Open visible coins of bombermanfield(17x17)
    - Perspective will be limited to 16x16 field in DQN
    - no wait and bombs
    - not yet used with a3c
3. `coinman2-v0`
    - Smaller version of `coinman-v0`  8x8
    - many coins
    - no wait and bombs
    - "local perspective" (render in 4 directions the visible coins for 4 fields; walls are not see through)
    - works well with a3c, even with structure
    - TODO: optionalize creation of structure
    - Terminates after 100 steps

4. `bombermandiehard-v0`
    - local perspective+ four corners if you wish to have them
    - player cannot be killed by own bombs
    - currently player is not utilizing bombs and collecting
    - Terminates after 200 steps

#### Rewards

Rewards are given differently for some actions, see table:

| Enviroment | Movement | correctly placing a bomb | Wrong input | surviving detonation | Getting killed | getting terminated |
| ------------- | ------------ | ----------------------------- | -------------- | ---------------------- | ---------------- | --------------------- |

WARNING: Not all configurations and envs are compatible. You have to adjust certain netrules (flatten parameters) to get a fair chance of good training

## Usage

1. A3C is executed e.g.: `python a3c_bomberman.py --algorithm a3c --save-dir train_with_4corners_bombermandiehard --train` train can be omitted to start test mode.
2.  DQN's are run like `python dqn_bomberman_4times5.py --mode=train`. Change mode to test to test.


Please refer to `A3C.md` and `DQN.md` for more information about these two algorithms or checkout the source code.
