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
    if args == (1, 2):
        return [[0.3], [-0.1]]  # weights
    if args == ():
        return 1   # bias


def test_dense_layer_forward(mocker):
    patched = mocker.patch('enzo.units.np.random.randn',
                           side_effect=patched_randn)
    layer = enzo.layers.DenseLayer(3)
    inputs = [[4, 5], [-2, 1.1], [0, 0.4]]
    assert layer.forward(inputs) == pytest.approx([1.7, 0.29, 0.96])
