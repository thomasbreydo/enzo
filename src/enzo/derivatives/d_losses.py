"""Derivative functions for loss functions."""

import numpy as np


def d_crossentropy(y_true, y_pred, epsilon=1e-12):
    r"""The derivative of the crossentropy loss function.

    Parameters
    ----------
    y_true : array_like
    y_pred : array_like
    epsilon : float
        The value at which `y_pred` is lower-bounded, by default 1e-12

    Returns
    -------
    array_like

    Notes
    -----
    The point of an `epsilon` (:math:`\epsilon`) is to allow the computation of
    :math:`\frac{y}{\hat{y}}` which is undefined at :math:`\hat{y}=0` by computing
    :math:`\frac{y}{\min(\hat{y}, \epsilon)}`. (Note: :math:`\hat{y}` is any value in
    `y_pred` and :math:`y` is any value in `y_true`).
    """
    return y_true / np.clip(y_pred, a_min=epsilon, a_max=None)
