import random
import numpy as np
import pytest
import enzo


@pytest.fixture
def mock_np_random_rand(monkeypatch):
    def new_rand(*args):
        return MOCKED_RANDOM_WEIGHTS[args]

    monkeypatch.setattr(np.random, "rand", new_rand)


def test_dense_layer_initializes_weights():
    for _ in range(5):
        n_units = random.randint(1, 5)
        input_length = random.randint(1, 5)
        layer = enzo.layers.DenseLayer(input_length, n_units)
        assert layer.weights.shape == (input_length + 1, n_units)


MOCKED_RANDOM_WEIGHTS = {(3, 2): np.array([[0.1, 0.5], [0.3, 0.0001], [0.2, 0.9]])}


def test_dense_layer_forward(mock_np_random_rand):
    layer = enzo.layers.DenseLayer(2, 2)
    samples = np.array([[0.2, 0.1], [0.9, 0.245], [0.9, 0.003]])
    layer.forward(samples)
    np.testing.assert_allclose(
        layer.outputs, [[0.25, 1.00001], [0.3635, 1.3500245], [0.2909, 1.3500003]]
    )


def test_append_column_of_ones_to_samples():
    samples = np.array([[1, 2], [3, 4], [5, 6]])
    with_col = enzo.layers.DenseLayer._append_column_of_ones(samples)
    assert (with_col == [[1, 2, 1], [3, 4, 1], [5, 6, 1]]).all()


def test_dense_layer_implicit_input_shape():
    pass  # TODO: move to test_models for Sequential
