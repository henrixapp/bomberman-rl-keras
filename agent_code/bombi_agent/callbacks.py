import numpy as np
from time import sleep
from settings import s

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

INPUT_SHAPE = (4,5)
WINDOW_LENGTH = 4

def setup(self):
    '''
    Called once in the beginning of the episode to setup bombi
    '''
    self.logger.info('Bombi awakes.')
    np.random.seed(123)
    nb_actions = 6
    
    #setting up network architecture
    self.model = Sequential([
        Flatten(input_shape=(WINDOW_LENGTH,4,5)),
        Dense(128),
        Activation("relu"),
        Dense(64),
        Activation("relu"),
        Dense(6),
        Activation("linear")
    ])
    
    #loading weights from file
    weights_filename = 'weights.h5f'
    self.model.load_weights(weights_filename)

def act(self):
    '''
    Called in every steps to determine the `next_action`
    '''
    self.logger.info('Bombi acts now.')
    
    #save perspective in memory 
    if self.game_state['step'] == 1:
        self.state = generate_state(perspective(self.game_state), 3)
    else:
        self.state = push_state(perspective(self.game_state), self.state)
        
    #predict next action 
    action = self.model.predict_on_batch(np.array([self.state])) 
    action = tf.nn.softmax(action)[0]
    action = np.argmax(K.eval(action))
    
    self.next_action = s.actions[action]  

def reward_update(self):
    pass

def end_of_episode(self):
    pass


def perspective(game_state, distance=5):
    '''
    Returns a 4 x distance matrix with the agent's view into each direction
    '''
    result = np.zeros((4,distance),dtype=np.int8) 
    x,y = game_state['self'][0],game_state['self'][1]
    arena = game_state['arena']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosions']
    k = 0
    for it_x, it_y in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        wand = False
        for i in range(distance): 
            #agents can't look through walls
            if(wand):
                result[k, i] = WALL
            else:
                if x+it_x*(i) < 0 or 0 > y+it_y*(i) or x+it_x*(i) > s.cols or s.rows < y+it_y*(i):
                    #outside of arena
                    wand = True
                    result[k, i] = WALL
                elif arena[x+it_x*(i), y+it_y*(i)] == WALL:
                    #walls
                    wand = True
                    result[k, i] = WALL
                else:
                    result[k,i] = arena[x+it_x*(i), y+it_y*(i)] 
                    #bombs
                    for b in bombs:
                        if b[0] == x+it_x*(i) and b[1] == y+it_y*(i):
                            result[k, i] = -2
                    #coins
                    for c in coins:
                        if c[0] == x+it_x*(i) and c[1] == y+it_y*(i):
                            result[k, i] = COIN 
                    
                    #explosions
                    if explosions[x+it_x*(i), y+it_y*(i)] > 0:
                        result[k,i] = EXPLOSION
        k = k+1
        
    #store if bomb is available
    result[0,0] = game_state['self'][3]
    
    return (result.astype('float32'))/7

def generate_state(state,length):
    '''
    creates a new memory of size `length` and fill it with the current `state`
    '''
    if length==0:
        return state
    return np.append([state],[state for i in range(length)], axis=0)
  
def push_state(state,current_state,length= WINDOW_LENGTH):
    '''
    adds the state to the current state
    '''
    if length==0:
        return state
    return np.append(current_state[1:(current_state.shape[0])],[state], axis=0)