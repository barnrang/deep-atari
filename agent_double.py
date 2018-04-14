from keras.models import Sequential, Model
from keras.layers import *
from fast_queue import fast_queue
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import random
import numpy as np
import gym
from collections import deque

class DQAgent:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.memory = fast_queue(size=5000)

    def _huber_loss(self, target, pred, clip_delta=1.):
        error = K.abs(target - pred)
        cond = error < clip_delta
        loss = tf.where(cond, 0.5 * K.square(error), clip_delta * (error - 0.5 * clip_delta))
        return loss


    def _build_model(self):
        frames_input = Input(self.config.img_size)
        normalized = Lambda(lambda x: x / 255.0)(frames_input)
        bottle_seq = [
            Conv2D(32,(7,7),padding='valid', activation='relu', input_shape=self.config.img_size),
            MaxPooling2D(),
            Conv2D(64,(5,5),padding='valid', activation='relu'),
            MaxPooling2D(),
            Conv2D(128,(3,3),padding='valid', activation='relu'),
            MaxPooling2D(),
            Flatten()
        ]

        action_seq = [
            Dense(256,activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(128, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(self.config.action_choices, activation='linear')
        ]
        tmp_model = Sequential(bottle_seq + action_seq)
        out = tmp_model(normalized)
        model = Model(input=frames_input, output=out)
        model.compile(optimizer=Adam(self.config.lr), loss=self._huber_loss)
        return model


    # Transform before passing images
    def gray_scale(self, images):
        return np.mean(images, axis=3).astype(np.uint8)

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
        batch_indice = self.memory.random_batch(batch_size)
        minibatch = [self.memory[x] for x in batch_indice]
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # Double Q-learning
                t_inner = self.model.predict(next_state)[0]
                t_score = self.target_model.predict(next_state)[0][np.argmax(t_inner)]
                target[0][action] = reward + self.config.gamma * t_score

            self.model.fit(state, target, epochs=1, verbose=0, callbacks=[self.memory])
            self.memory.save_loss(batch_indice[idx])
        if self.config.epsilon > self.config.epsilon_min:
            self.config.epsilon *= self.config.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)