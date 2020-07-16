import numpy as np
from . import activation as act


class Neuron:
    def __init__(self, weights=None, bias=None, activation=None):
        if activation is None:
            activation = act.relu
        self.weights = weights
        self.bias = bias
        self.activation = activation

    # TODO: predict
