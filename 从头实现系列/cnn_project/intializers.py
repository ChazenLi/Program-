import numpy as np

class XavierUniformInit:
    def __call__(self, shape):
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)

class ZerosInit:
    def __call__(self, shape):
        return np.zeros(shape)
