import os
import numpy as np


def load_mnist(batch_size=1, is_training=True):
    path = '../data'
    fd = open(os.path.join(path, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trainX = loaded[16:].reshape((60000, 28, 28)).astype(np.float32)

    fd = open(os.path.join(path, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trainY = loaded[8:].reshape((60000)).astype(np.int32)

    trX = trainX[:50000] / 255.
    trY = trainY[:50000]

    valX = trainX[50000:, ] / 255.
    valY = trainY[50000:]
    return trX, trY, valX, valY