""""Derivative functions for activation functions."""

import numpy as np


def ddx_noactivation(rows):
    """The derivative of a f(x)=x activation (*noactivation*).

    Parameters
    ----------
    rows : array_like

    Returns
    -------
    array_like
        The derivative evaluated at each row in `rows`.
    """
    rows = np.asarray(rows)
    return np.ones(rows.shape)


def ddx_relu(rows):
    r"""The derivative of the rectified linear unit (*relu*).

    Parameters
    ----------
    rows : array_like

    Returns
    -------
    array_like
        The derivative evaluated at each row in `rows`.

    Notes
    -----
    :func:`ddx_relu` evaluated at 0 is 0 despite the fact that the true derivative of
    ReLU evaluated at 0 is undefined. This allows for a continuous derivative
    function, letting weights set to 0 to have a derivative.

    .. math:: \frac{dr}{dx}\Bigr|_0 = 0
    """
    rows = np.asarray(rows)
    return rows > 0


def ddx_sigmoid(rows):
    """The derivative of the sigmoid activation function.

    Parameters
    ----------
    rows : array_like

    Returns
    -------
    array_like
        The derivative evaluated at each row in `rows`.
    """
    rows = np.asarray(rows)
    return np.exp(-rows) / (1 + np.exp(-rows)) ** 2


# TODO
def _ddx_softmax(row):
    return -np.exp(row) / row.sum() ** 2


# TODO
def ddx_softmax(rows):
    """The derivative of the softmax activation function.

    Parameters
    ----------
    rows : array_like

    Returns
    -------
    array_like
        The derivative evaluated at each row in `rows`.
    """
    rows = np.asarray(rows)
    return np.apply_along_axis(_ddx_softmax, arr=rows, axis=1)
