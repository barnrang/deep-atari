import numpy as np
import random
import keras
eps = 10e-8

class fast_queue(keras.callbacks.Callback):
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.sam_loss = [0] * (size + 1)
        self.indice = list(range(0, size + 1))
        
        # state index (training)
        self.idx = 0


        self.size = size + 1
        self.start = 0
        self.end = 0

    def __get__(self, obj, objtype):
        if self.start <= self.end:
            print([self[x] for x in range(self.start, self.end)])
        else:
            print([self[x] for x in range(self.start, self.end + self.size)])

    def __len__(self):
        if self.start <= self.end:
            return self.end - self.start
        else:
            return self.size + self.end - self.start

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
        self.sam_loss[self.end] = eps
        self.end = (self.end + 1) % self.size

        if self.start == self.end:
            self.sam_loss[self.start] = 0
            self.start = (self.start + 1) % self.size

if __name__ == '__main__':
    test = fast_queue(10)
    for i in range(6):
        test.append(i)
    for i in range(5):
        test.append(i)
    print(test)
    print(len(test))
    print((test.start, test.end))  