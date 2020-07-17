'''Activation functions'''

import numpy as np


def relu(n):
    '''Return max(0, n).'''
    return n if n > 0 else 0


def sigmoid(n):
    '''Return 1 / (1 + e ^ -n).'''
    return 1 / (1 + np.exp(-n))


def noactivation(x):
    return x


def softmax():
    pass  # TODO
