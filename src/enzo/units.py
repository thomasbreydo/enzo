import numpy as np
from . import activation as act


class Neuron:
    def __init__(self, weights=None, bias=None, activation=None):
        self.weights = weights
        self.bias = bias
        if activation is None:
            self.activation = act.noactivation
        else:
            self.activation = activation

    def process(self, inp):
        if self.weights is None:
            self.weights = np.random.randn(1, len(inp))
        if self.bias is None:
            self.bias = np.random.randn()
        try:
            pre_activation = inp @ self.weights + self.bias
        except TypeError:  # inp not np.array
            pre_activation = np.array(inp) @ self.weights + self.bias
        return self.activation(pre_activation)


class Perceptron:
    pass
