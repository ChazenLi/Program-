class Layer(object):
    def __init__(self, name):
        self.name = name
        self.params = None
        self.grads = None
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
