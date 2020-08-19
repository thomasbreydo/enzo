import pytest
import numpy as np
from enzo import losses


def test_crossentropy():
    y_true = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    y_pred = [
        [0.9, 0, 0, 0.1],
        [0.3, 0.5, 0.15, 0.05],
        [0.25, 0.5, 0.2, 0.05],
        [0.1, 0.1, 0.1, 0.7],
    ]
    np.testing.assert_allclose(losses.crossentropy(y_true, y_pred), 0.58978885)
