from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import random
import numpy as np
import gym
from collections import deque
from agent import DQAgent

class Config:
    img_size = (210, 160, 4)
    dropout_rate = 0.75
    lr = 3e-5
    action_choices = 4
    epsilon = 0.
    epsilon_min = 0.1
    epsilon_decay = 0.99
    gamma = 0.95


def main():
    config = Config()
    agent = DQAgent(config)
    agent.load_model('model/atariv2.h5')
    env = gym.make('Breakout-v0')
    observation = env.reset()
    state = np.expand_dims(observation, axis=0)
    state = agent.gray_scale(state)
    state = np.stack([state for _ in range(4)], axis=-1)
    done = False
    tot_reward = 0.0
    while not done:
        env.render()
        Q = agent.model.predict(state)[0]
        action = np.argmax(Q)
        observation, reward, done, info = env.step(action)
        tmp_state = agent.gray_scale(np.expand_dims(observation, axis=0))
        state[:,:,:,0:3] = state[:,:,:,1:4]
        state[:,:,:,3] = tmp_state
        tot_reward += reward
    print('Game ended! Total reward: {}'.format(tot_reward))

if __name__ == "__main__":
    main()
