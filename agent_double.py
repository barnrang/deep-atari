from keras.models import Sequential, Model
from keras.layers import *
from fast_queue import fast_queue
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import tensorflow as tf
import random
import numpy as np
import gym
from collections import deque

class Config:
    img_size = (205, 80, 4)
    dropout_rate = 0.1
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
        print(self.model.summary())
        self.memory = fast_queue(size=60000)

    def _huber_loss(self, target, pred, clip_delta=1.):
        error = K.abs(target - pred)
        cond = error < clip_delta
        loss = tf.where(cond, 0.5 * K.square(error), clip_delta * (error - 0.5 * clip_delta))
        return loss


    def _build_model(self):
        frames_input = Input(self.config.img_size)
        normalized = Lambda(lambda x: x / 255.0, input_shape=self.config.img_size)
        bottle_seq = [
            normalized,
            Conv2D(32,(7,7),padding='valid', activation='relu'),
            MaxPooling2D(),
            Conv2D(64,(5,5),padding='valid', activation='relu'),
            MaxPooling2D(),
            Conv2D(128,(3,3),padding='valid', activation='relu'),
            MaxPooling2D(),
            Flatten()
        ]

        action_seq = [
            Dense(256,activation='relu'),
            #Dropout(self.config.dropout_rate),
            Dense(128, activation='relu'),
            #Dropout(self.config.dropout_rate),
            Dense(self.config.action_choices, activation='linear')
        ]
        model = Sequential(bottle_seq + action_seq)
        #out = tmp_model(normalized)
        #model = Model(input=frames_input, output=out)
        model.compile(optimizer=RMSprop(lr=0.00003, rho=0.95, epsilon=0.01), loss=self._huber_loss)
        return model


    # Transform before passing images
    def gray_scale(self, images):
        #print(images.shape)
        images = images[:,::2,::2,:,:]
        return np.mean(images, axis=3).astype(np.uint8)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if random.uniform(0, 1) < self.config.epsilon:
            return random.randrange(self.config.action_choices)
        else:
            return np.argmax(self.model.predict(self.gray_scale(state))[0])


    def replay(self, batch_size):
        if batch_size > len(self.memory):
            return
        
        if random.random() < 0.5:    
            batch_indice = self.memory.random_batch(batch_size)
        else:
            batch_indice = self.memory.random_unweight_batch(batch_size)
        print(batch_indice)
        minibatch = [self.memory.data[x] for x in batch_indice]
        loss_batch = [self.memory.sam_loss[x] for x in batch_indice]
        print(loss_batch)
        prev = None
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            #if prev is not None:
            #    print(np.sum(prev - state))
            prev = state.copy()
            target = self.model.predict(state)
            #print(target)
            if done:
                target[0][action] = reward
            else:
                # Double Q-learning
                t_inner = self.model.predict(next_state)[0]
                t_score = self.target_model.predict(next_state)[0][np.argmax(t_inner)]
                target[0][action] = reward + self.config.gamma * t_score


            self.model.fit(state, target, epochs=1, verbose=0, callbacks=[self.memory])
            self.memory.save_loss(batch_indice[idx])
        '''states = np.zeros((batch_size,) + (self.config.img_size))
        actions = np.zeros(batch_size).astype(np.int)
        rewards = np.zeros(batch_size)
        next_states =  np.zeros((batch_size,) + (self.config.img_size))
        dones = np.zeros(batch_size)
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[idx] = state
            actions[idx] = action
            rewards[idx] = reward
            next_states[idx] = next_state
            dones[idx] = done
        target = self.model.predict(states)
        t_inner = self.model.predict(next_states)
        t_score = self.target_model.predict(next_states)
        t_max = np.argmax(t_inner, axis=1)
        t_score = np.choose(t_max, t_score.T)
        print(t_score)
        print(rewards)
        print(target.shape)
        print(actions)
        target[list(range(batch_size)), actions] = rewards + self.config.gamma * t_score

        self.model.fit(states, target, epochs=1, verbose=0, callbacks=[self.memory])
        self.memory.save_loss(batch_indice)'''

        if self.config.epsilon > self.config.epsilon_min:
            self.config.epsilon *= self.config.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((self.gray_scale(state), action, reward, self.gray_scale(next_state), done))

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)


if __name__ == "__main__":
    config = Config()
    test_agent = DQAgent(config)
    test_agent
