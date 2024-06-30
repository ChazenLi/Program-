import numpy as np
from layers.base import Layer

class Activation(Layer):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)
    
    def backward(self, grad):
        return self.derivative_func(self.inputs) * grad
    
    def func(self, x):
        raise NotImplementedError
    
    def derivative_func(self, x):
        raise NotImplementedError

class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU")

    def func(self, x):
        return np.maximum(x, 0.0)
    
    def derivative_func(self, x):
        return x > 0.0
