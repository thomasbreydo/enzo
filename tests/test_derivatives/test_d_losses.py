import pytest
import numpy as np
from enzo import derivatives


def test_d_crossentropy():
    y_true = [[0, 0, 1, 0], [1, 0, 0, 0]]
    y_pred = [[0.1, 0.2, 0.6, 0.1], [0.005, 0.1, 0.1, 0.795]]
    np.testing.assert_allclose(
        derivatives.d_crossentropy(y_true, y_pred), [[0, 0, -5 / 3, 0], [-200, 0, 0, 0]]
    )
