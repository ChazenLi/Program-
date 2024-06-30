import numpy as np

class BaseLoss(object):
    def loss(self, predicted, actual):
        raise NotImplementedError
    
    def grad(self, predicted, actual):
        raise NotImplementedError

class CrossEntropyLoss(BaseLoss):
    def loss(self, predicted, actual):
        m = predicted.shape[0]
        exps = np.exp(predicted - np.max(predicted, axis=1, keepdims=True))
        p = exps / np.sum(exps, axis=1, keepdims=True)
        nll = -np.log(np.sum(p * actual, axis=1))
        return np.sum(nll) / m
    
    def grad(self, predicted, actual):
        m = predicted.shape[0]
        grad = predicted - actual
        return grad / m
