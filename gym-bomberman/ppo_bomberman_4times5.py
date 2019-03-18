# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

import numpy as np

import gym
import gym_bomberman

from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam

import numba as nb
from tensorboardX import SummaryWriter

ENV = 'bombermandiehard-v0'

EPISODES = 1000#000

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
NOISE = 1.0 # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 256
BATCH_SIZE = 64
NUM_ACTIONS = 6
#NUM_STATE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 1e-3
LR = 1e-4 # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10)))
    return loss


class Agent:
    def __init__(self):
        self.env = gym.make(ENV)
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.action_size = self.env.action_space.n
        self.state_size = 20 #self.env.observation_space.shape
        self.critic = self.build_critic()
        self.actor = self.build_actor()
        self.episode = 0
        self.observation = self.env.reset()
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.name = self.get_name()
        self.writer = SummaryWriter(self.name)
        self.gradient_steps = 0

    def get_name(self):
        name = 'AllRuns/'
        name += 'discrete/'
        name += ENV
        return name


    def build_actor(self):
        state_input = Input(shape=(self.state_size,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_critic(self):

        state_input = Input(shape=(self.state_size,))
        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        if self.episode % 100 == 0:
            self.val = True
        else:
            self.val = False
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, self.state_size), DUMMY_VALUE, DUMMY_ACTION])
        #p = p[:, 0]
        if self.val is False:
            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        return action, action_matrix, p


    def transform_reward(self):
        if self.val is True:
            self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            action, action_matrix, predicted_action = self.get_action()
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict(obs.reshape(BUFFER_SIZE, self.state_size))
            #print("reward shape=", obs.shape, "pred_values shape= ",pred_values.shape)
            advantage = reward - pred_values
            # advantage = (advantage - advantage.mean()) / advantage.std()
            #print("obs",obs.shape, "advantage", advantage.shape, "old_pred", old_prediction.shape, "action",action.shape)
            actor_loss = self.actor.fit([obs.reshape(BUFFER_SIZE, self.state_size), advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            critic_loss = self.critic.fit([obs.reshape(BUFFER_SIZE, self.state_size)], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
            if self.episode % 10 == 0:
                print("Episode: ", self.episode, " Actor loss: ",  actor_loss.history['loss'][-1], " Critic loss: ", critic_loss.history['loss'][-1], " reward: ", reward[-1])
            self.gradient_steps += 1

    def play(self):
        self.actor.test(self.env, nb_episodes=4, visualize=True)

if __name__ == '__main__':
    ag = Agent()
    ag.run()
    