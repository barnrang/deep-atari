from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import random
import numpy as np
import gym
from collections import deque

EP = 5000

class Config:
    img_size = (210, 160, 3)
    dropout_rate = 0.75
    lr = 3e-5
    action_choices = 3
    epsilon = 1.
    epsilon_min = 0.1
    epsilon_decay = 0.99
    gamma = 0.95


class DQAgent:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.memory = deque(maxlen=2000)

    def _huber_loss(self, target, pred, clip_delta=1.):
        error = K.abs(target - pred)
        cond = error < clip_delta
        loss = tf.where(cond, 0.5 * K.square(error), clip_delta * (error - 0.5 * clip_delta))
        return K.mean(loss)

    def _build_model(self):
        bottle_seq = [
            Conv2D(32,(5,5),padding='valid', activation='relu', input_shape=self.config.img_size),
            MaxPooling2D(),
            Conv2D(64,(3,3),padding='valid', activation='relu'),
            MaxPooling2D(),
            Conv2D(128,(3,3),padding='valid', activation='relu'),
            MaxPooling2D(),
            Flatten()
        ]

        action_seq = [
            Dense(128,activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(64, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(self.config.action_choices, activation='linear')
        ]
        model = Sequential(bottle_seq + action_seq)
        model.compile(optimizer=Adam(self.config.lr), loss=self._huber_loss)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if random.uniform(0, 1) < self.config.epsilon:
            return random.randrange(self.config.action_choices)
        else:
            return np.argmax(self.model.predict(state)[0])


    def replay(self, batch_size):
        if batch_size > len(self.memory):
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.config.gamma * np.max(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.config.epsilon > self.config.epsilon_min:
            self.config.epsilon *= self.config.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)


def main():
    env = gym.make('Breakout-v0')
    state_space = env.observation_space.shape
    action_size = env.action_space.n
    config = Config()
    config.action_choices = action_size
    config.img_size = state_space
    agent = DQAgent(config)
    agent.load_model('model/atariv1.h5')
    done = False
    batch_size = 32

    for e in range(EP):
        state = env.reset()
        state = state.reshape((1,) + state_space)
        point = 0
        for t in range(5000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward  = reward if not done else -10
            point += reward
            next_state = next_state.reshape((1,) + state_space)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print('episode {}/{}, score: {}'.format(e, EP, point))
                break
        agent.replay(batch_size)
        if e % 100 == 0:
            agent.save_model('model/atariv1.h5')

if __name__ == "__main__":
    main()
