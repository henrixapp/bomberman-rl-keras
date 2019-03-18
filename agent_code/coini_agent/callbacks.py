
import numpy as np
from time import sleep
import gym
import gym_bomberman
from settings import s

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

tf.enable_eager_execution()

EXPLOSION = -3
BOMB = -2
WALL = -1
FREE = 0
CRATE = 1
COIN = 2
PLAYER = 3

class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    print((self.state_size,self.action_size))
    self.flatten0 = layers.Flatten(input_shape=(500+1,4,4))
    self.dense1 = layers.Dense(128)
    self.dense1a = layers.Dense(64, activation='relu')
    self.activation1 = layers.Activation('relu')
    self.policy_logits = layers.Dense(action_size)
    self.dense2 = layers.Dense(256, activation='relu')
    self.values =layers.Dense(1)

  def call(self, inputs):
    # Forward pass
    z= self.flatten0(inputs)
    g = self.dense1(z)
    x = self.dense1a(self.activation1(g))
    logits = self.policy_logits(x)
    v1 = self.dense2(z)
    values = self.values(v1)
    return logits, values
'''
def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
  """Helper function to store score and print statistics.

  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
  """
  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  if(episode%10==0):
    print(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
  result_queue.put(global_ep_reward)
  return global_ep_reward
'''


def setup(self):
    self.logger.info('Coini awakes.')
    self.env = gym.make('coinman2-v0').unwrapped
    self.state = self.env.reset()
    self.state_size = self.env.observation_space.shape[0]*self.env.observation_space.shape[1]
    self.action_size = self.env.action_space.n
    self.model = ActorCriticModel(self.state_size, self.action_size)
    print(self.model(tf.convert_to_tensor(self.state[None, :], dtype=tf.float32)))
    self.model.load_weights('agent_code/coini_agent/weights.h5')
    
def act(self):
    self.logger.info('Coini acts now.')
    #self.env.render(mode='human')
    #self.state = self.env._get_obs()
    policy, value = self.model(tf.convert_to_tensor(perspective(self.game_state)[None,:], dtype=tf.float32))
    #policy = tf.nn.softmax(policy)
    action = np.argmax(policy)
    
    #if action < 4:
    #    action += 3
    #    action = action % 4
    if action == 0: 
        action = 0
    elif action == 1:
        action = 1
    elif action == 2: 
        action = 2
    elif action == 3:
        action = 3
    print(action)
    #self.state, reward, done, _ = self.env.step(action)
        
    self.next_action = s.actions[action]
    

def reward_update(self):
    pass

def end_of_episode(self):
    pass

def perspective(game_state, distance=4):# added own field
    result = np.zeros((4,distance))
    x = game_state['self'][0]
    y = game_state['self'][1]
    k = 0
    for it_x,it_y in [(-1,0),(1,0),(0,1),(0,-1)]:
        wand = False
        for i in range(distance):# should we be able to look over walls? --> currently not
            if(wand):
                result[k,i]= WALL
            else:
                if x+it_x*(i+1)<0 or 0 >y+it_y*(i+1) or x+it_x*(i+1)>7 or 7 < y+it_y*(i+1):# TODO; Wand bedingung updaten
                    wand= True
                    result[k,i]=WALL
                elif game_state['arena'][x+it_x*(i+1),y+it_y*(i+1)] == WALL:
                    wand= True
                    result[k,i]= WALL
                else:
                    for b in game_state['bombs']:
                        if b[0] == x+it_x*(i+1) and b[1] == y+it_y*(i+1):
                            result[k,i] = BOMB
                    for c in game_state['coins']:
                        if c[0] == x+it_x*(i+1) and c[1] == y+it_y*(i+1):
                            result[k,i] = COIN # TODO Players
        k = k+1
    return result
