import pytest
import numpy as np
from enzo.models import Model
from enzo.layers import DenseLayer
from enzo.exceptions import LayerBuildingError
from mockers import mock_np_random_rand
from mockers import MOCKED_RANDOM_WEIGHTS


def test_model_without_explicit_first_layer_input_fails():
    with pytest.raises(LayerBuildingError, match="first layer"):
        Model([DenseLayer(16), DenseLayer(16), DenseLayer(16)])


MOCKED_RANDOM_WEIGHTS[(3, 3)] = [
    [-0.31, -0.97, -0.95],
    [-0.25, 0.8, 0.28],
    [0.8, -0.29, -0.43],
]
MOCKED_RANDOM_WEIGHTS[(4, 2)] = [
    [0.9, 0.5],
    [-0.55, -0.77],
    [0.17, -0.04],
    [0.18, -0.1],
]


def test_model_forward(mock_np_random_rand):
    model = Model([DenseLayer(3, input_length=2), DenseLayer(2)])
    model.forward([[0.89, 0.5], [-0.15, -0.97]])
    np.testing.assert_allclose(model.outputs, [[0.53919, 0.09955], [1.1601, 0.4445]])


def test_model_forward_with_1d_samples(mock_np_random_rand):
    model = Model([DenseLayer(3, input_length=2), DenseLayer(2)])
    model.forward([-0.15, -0.97])
    np.testing.assert_allclose(model.outputs, [[1.1601, 0.4445]])
