import random
import pytest
import numpy as np
import enzo
from mockers import mock_np_random_rand
from mockers import MOCKED_RANDOM_WEIGHTS


def test_dense_layer_initializes_weights():
    for _ in range(5):
        n_units = random.randint(1, 5)
        input_length = random.randint(1, 5)
        layer = enzo.layers.DenseLayer(n_units, input_length=input_length)
        layer.build()
        assert layer.weights.shape == (input_length + 1, n_units)


MOCKED_RANDOM_WEIGHTS[(3, 2)] = np.array([[0.1, 0.5], [0.3, 0.0001], [0.2, 0.9]])


def test_dense_layer_forward(mock_np_random_rand):
    layer = enzo.layers.DenseLayer(2, input_length=2)
    samples = np.array([[0.2, 0.1], [0.9, 0.245], [0.9, 0.003]])
    layer.build()
    layer.forward(samples)
    np.testing.assert_allclose(
        layer.outputs, [[0.25, 1.00001], [0.3635, 1.3500245], [0.2909, 1.3500003]]
    )


MOCKED_RANDOM_WEIGHTS[(5, 2)] = np.array(
    [[0.4, 0.245], [-0.2, 0.0032], [0.3, -0.88], [0.7, 0.9], [-0.1, -0.2]]
)


def test_dense_layer_forward_with_1d_samples(mock_np_random_rand):
    layer = enzo.layers.DenseLayer(2, input_length=4)
    samples = np.array([0.2, 0.1, 0.003, 0.9])
    layer.build()
    layer.forward(samples)
    np.testing.assert_allclose(layer.outputs, [[0.5909, 0.65668]])
