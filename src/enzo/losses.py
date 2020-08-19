"""Loss functions."""

import numpy as np


def crossentropy(y_true, y_pred, epsilon=1e-12):
    r"""Calculate the crossentropy loss of `y_true` with respect to `y_pred`.

    Parameters
    ----------
    y_true : array_like
        One-hot encoded ture labels.
    y_pred : array_like
        Model predictions.
    epsilon : float, optional
        The value at which `y_pred` is lower-bounded, by default 1e-12

    Notes
    -----
    The point of an `epsilon` (:math:`\epsilon`) is to allow the computation of
    :math:`\log(\hat{y})` which is undefined at :math:`\hat{y}=0` by computing
    :math:`\log(\min(\hat{y}, \epsilon))`. (Note: :math:`\hat{y}` is any value in
    `y_pred`).
    """
    by_row = -(y_true * np.log(np.clip(y_pred, a_min=epsilon, a_max=None)))
    try:
    return by_row.sum(axis=1).mean()
    except np.AxisError:
        return by_row.sum(axis=0)
