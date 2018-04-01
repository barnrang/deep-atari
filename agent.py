from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K
import numpy as np


config = {
    img_size = (210, 160, 3),
    dropout_rate = 0.5,
    action_choices
}

class agent:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()

    def replay(self):

    def _huber_loss(self, target, pred, clip_delta=1.):
        error = K.abs(target - pred)
        cond = error < clip_delta
        loss = tf.cond(cond, lambda: 0.5 * K.square(error), lambda: clip_delta * (error - 0.5 * clip_delta))
        return K.mean(loss)

    def _build_model(self):
        bottle_seq = [
            Conv2D(32,(3,3),padding='valid', activation='relu', input_shape=self.config.img_size),
            MaxPooling2D(),
            Conv2D(64,(3,3),padding='valid', activation='relu'),
            MaxPooling2D(),
            Flatten()
        ]

        action_seq = [
            Dense(64,activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(32, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(self.config.action_choices, activation='softmax')
        ]
        model = Sequential(bottle_seq + action_seq)
        return model

