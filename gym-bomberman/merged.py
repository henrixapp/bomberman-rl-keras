from __future__ import division
import argparse
import gym
import gym_bomberman
from PIL import Image
import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import keras
import keras.callbacks
from keras.layers.merge import concatenate
from keras.utils import plot_model

RENDER_CORNERS = False
RENDER_HISTORY = False
INPUT_SHAPE = (5+RENDER_CORNERS+RENDER_HISTORY, 5)
WINDOW_LENGTH = 1
input_shape =  (WINDOW_LENGTH,) + INPUT_SHAPE

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='bombermandiehard-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print(nb_actions)

#input layer
visible = Input(shape=input_shape)

#a3c layers
z = Flatten(input_shape=(input_shape))(visible)
g = Dense(128, activation='relu')(z)
x = Dense(64, activation='relu')(g)
#logits = layers.Dense(6)
#v1 = Dense(256, activation='relu')(z)
#values = Dense(1)(v1)

#dqn layers
flattened = Flatten(input_shape=(input_shape))(visible)
dense1 = Dense(128, activation='relu')(flattened)
dense2 = Dense(64, activation='relu')(dense1)
#dense3 = Dense(6, activation='linear')(dense2)
            
#merge layer
merge = concatenate([x, dense2])

#interpretation layer
hidden = Dense(12, activation='relu')(merge)

#prediction output
output = Dense(6, activation='sigmoid')(hidden)
model = Model(inputs=visible, outputs=output)

print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=150000, visualize=False, verbose=1)


weights_filename = 'merged_{}_weights.h5f'.format(args.env_name)
dqn.save_weights(weights_filename, overwrite=True)
dqn.test(env, nb_episodes=50, visualize=True)