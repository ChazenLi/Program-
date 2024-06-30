import numpy as np
from layers.base import Layer

class XavierUniformInit:
    def __call__(self, shape):
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)

class ZerosInit:
    def __call__(self, shape):
        return np.zeros(shape)


class Dense(Layer):
    def __init__(self, num_in, num_out, w_init=XavierUniformInit(), b_init=ZerosInit()):
        super().__init__("Linear")
        self.params = {
            "w": w_init([num_in, num_out]),
            "b": b_init([1, num_out])
        }
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
    
    def backward(self, grad):
        self.grads = {
            "w": self.inputs.T @ grad,
            "b": np.sum(grad, axis=0)
        }
        return grad @ self.params["w"].T
