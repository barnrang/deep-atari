import numpy as np
import random
import keras
eps = 10e-8

class fast_queue(keras.callbacks.Callback):
    def __init__(self, size):
        self.data = [None] * size
        self.sam_loss = [eps] * size
        self.indice = list(range(0, size))
        
        # state index (training)
        self.idx = 0


        self.size = size
        self.start = 0
        self.end = 0

    def __len__(self):
        if self.start <= self.end:
            return self.end - self.start
        else:
            return self.start + self.end - 1

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % self.size]

    def random_batch(self, sample_num):
        # Return index list
        # Prioritize higher loss sample (Prioritized Replay)
        return random.choices(self.indice, self.sam_loss, k=sample_num)
    
    def on_batch_end(self, batch, logs={}):
        # Keras callback
        self.loss = logs.get('loss')

    def save_loss(self, idx):
        self.sam_loss[idx] = self.loss

    def append(self, item):
        self.data[self.end] = item
        self.sam_loss[self.end] = 0
        self.end = (self.end + 1) % self.size

        if self.start == self.end:
            self.start = (self.start + 1) % self.size

    