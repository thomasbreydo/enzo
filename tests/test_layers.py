import pytest
import enzo


def test_dense_layer_init():
    layer = enzo.layers.DenseLayer(20)
    for i, neuron in enumerate(layer.neurons):
        assert isinstance(neuron, enzo.units.Neuron)
        assert neuron.activation == enzo.activation.noactivation
    assert i == 19, 'wrong number of neurons made'
    layer = enzo.layers.DenseLayer(20, activation=enzo.activation.relu)
    assert layer.neurons[0].activation == enzo.activation.relu


def patched_randn(*args):
    if args == (2, 1):
        return [[0.3], [-0.1]]  # weights
    if args == ():
        return 1   # bias
    raise ValueError('unexpected args passed')


def test_dense_layer_forward(mocker):
    patched = mocker.patch('enzo.units.np.random.randn',
                           side_effect=patched_randn)
    layer = enzo.layers.DenseLayer(3)
    inputs = [[1, 2], [-0.5, 0.1]]
    assert layer.forward(inputs) == [[pytest.approx(1.1)] * 3,
                                     [pytest.approx(0.84)] * 3]
