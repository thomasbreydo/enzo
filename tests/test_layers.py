import pytest
import enzo


def test_dense_layer_init():
    layer = enzo.layers.DenseLayer(20)
    for i, neuron in enumerate(layer.neurons):
        assert isinstance(neuron, enzo.units.Neuron)
        assert neuron.activation == enzo.activation.noactivation
    assert i == 19, 'wrong number of neurons made'
