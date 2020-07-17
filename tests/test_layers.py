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
    if args == (3, 1):
        return [[0], [0], [-1]]  # weights for layer 2
    if args == (2, 1):
        return [[0.3], [-0.1]]  # weights for layer 1
    if args == ():
        return 1   # bias
    raise ValueError('unexpected args passed')


def test_dense_layer_forward(mocker):
    patched = mocker.patch('enzo.units.np.random.randn',
                           side_effect=patched_randn)
    layer1 = enzo.layers.DenseLayer(3)
    layer2 = enzo.layers.DenseLayer(2)
    inputs = [[1, 2], [-0.5, 0.1]]
    after_layer1 = layer1.forward(inputs)
    assert after_layer1 == [[pytest.approx(1.1)] * 3,
                            [pytest.approx(0.84)] * 3]
    after_layer2 = layer2.forward(after_layer1)
    assert after_layer2 == [[pytest.approx(-0.1)] * 2,
                            [pytest.approx(0.16)] * 2]
