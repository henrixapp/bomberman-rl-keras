
import numpy as np
from time import sleep
import gym
import gym_bomberman
from settings import s

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

EXPLOSION = -3
WALL = -1
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3

RENDER_CORNERS = False
RENDER_HISTORY = False
INPUT_SHAPE = (4+RENDER_CORNERS+RENDER_HISTORY, 5)
WINDOW_LENGTH = 4

def setup(self):
    self.logger.info('Bombi awakes.')
    self.env = gym.make('bombermandiehard-v0')
    np.random.seed(123)
    self.env.seed(123)
    nb_actions = self.env.action_space.n
    print(nb_actions)
    self.model = Sequential([
        Flatten(input_shape=(WINDOW_LENGTH,4+RENDER_CORNERS+RENDER_HISTORY, 5)),
        Dense(128),
        Activation("relu"),
        Dense(64),
        Activation("relu"),
        Dense(6),
        Activation("linear")
    ])
    print(self.model.summary())
    #memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    #policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=12500000)
    #self.dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=50000, gamma=.99, target_model_update=5000, train_interval=4, delta_clip=1.)
    #self.dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,gamma=.99)
    #self.dqn.compile(Adam(lr=.00025), metrics=['mae'])
    weights_filename = 'agent_code/bombi_agent/weights.h5f'
    self.model.load_weights(weights_filename)

    

def act(self):
    self.logger.info('Bombi acts now.')
    #print(arena)
    #setup memory
    if self.game_state['step'] == 1:
        self.state = generate_state(perspective(self.game_state), 3)
    else:
        self.state = push_state(perspective(self.game_state), self.state)
    #print(self.state)
    observation = perspective(self.game_state)
    print(observation)
    #action = self.dqn.forward(observation)
    
    action = self.model.predict_on_batch(np.array([self.state[::-1]])) #.flatten() #[::-1]
    action = tf.nn.softmax(action)[0]
    print(K.eval(action))
    action = np.argmax(K.eval(action))
    #print(value, action)
    reward = -1
    if action == 5:
        reward = 2
    #self.dqn.backward(reward, False)
    #print(self.events)
    #print(perspective(self.game_state))
    print(action)
    #if action < 4:
    #    action += 3
    #    action = action % 4
    #if action == 0: 
    #    action = 0
    #elif action == 1:
    #    action = 1
    #elif action == 2: 
    #    action = 2
    #elif action == 3:
    #    action = 3
    
    self.env.step(action)
    #self.env.render()

    self.next_action = s.actions[action]
    

def reward_update(self):
    pass

def end_of_episode(self):
    pass


def perspective(game_state, distance=5):# added own field
    result = np.zeros((4+RENDER_CORNERS+ RENDER_HISTORY, distance),dtype=np.int8) #ToDo: case for rendercorners needs to be implemented
    x,y = game_state['self'][0],game_state['self'][1]
    arena = game_state['arena']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosions']
    k = 0
    for it_x, it_y in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        wand = False
        for i in range(distance):  # should we be able to look over walls? --> currently not
            if(wand):
                result[k, i] = WALL
            else:
                # TODO; Wand bedingung updaten
                if x+it_x*(i) < 0 or 0 > y+it_y*(i) or x+it_x*(i) > s.cols or s.rows < y+it_y*(i):
                    wand = True
                    result[k, i] = WALL
                elif arena[x+it_x*(i), y+it_y*(i)] == WALL:
                    wand = True
                    result[k, i] = WALL
                else:
                    result[k,i] = arena[x+it_x*(i), y+it_y*(i)] # forgotten first important!
                    for b in bombs:
                        if b[0] == x+it_x*(i) and b[1] == y+it_y*(i):
                            result[k, i] = -2
                    for c in coins:
                        if c[0] == x+it_x*(i) and c[1] == y+it_y*(i):
                            result[k, i] = COIN  # TODO Players, Explosions
                    if explosions[x+it_x*(i), y+it_y*(i)] > 0:
                        print("Explosion BOOOOOOOM")
                        result[k,i] = EXPLOSION
        k = k+1
    result[0,0] = game_state['self'][3]
    return (result.astype('float32'))/7

def push_state(state,current_state,length= WINDOW_LENGTH):
  if length==0:
    return state
  return np.append(current_state[1:(current_state.shape[0])],[state], axis=0)
def generate_state(state,length):
  if length==0:
    return state
  return np.append([state],[state for i in range(length)], axis=0)