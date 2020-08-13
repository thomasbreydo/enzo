"""Activation functions"""

import numpy as np


def relu(matrix):
    """Apply max(0, n) to each n in `matrix`.

    Parameters
    ----------
    matrix : `list` of `list`
    """
    return matrix * (matrix > 0)


def sigmoid(matrix):
    """Apply 1 / (1 + e ^ -n) to each n in `matrix`.

    Parameters
    ----------
    matrix : `list` of `list`
    """
    return 1 / (1 + np.exp(-matrix))


def noactivation(matrix):
    """Do nothing, return `matrix`."""
    return matrix


def _softmax(row):
    raised_to_the_e = np.exp(row)
    return raised_to_the_e / sum(raised_to_the_e)


def softmax(matrix):
    """Perform softmax scaling for each row in `matrix`."""
    return np.apply_along_axis(_softmax, 1, matrix)
