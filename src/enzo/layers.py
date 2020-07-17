from .units import Neuron


class DenseLayer:
    '''TODO: add desc

    Parameters
    ----------
    units : int
        The number of `class:Neuron` instances to add to this layer.
    weights : list of numeric
        The initial list of weights for all neurons.
    bias : numeric
        The initial bias for all neurons.
    activation : callable
        The activation function for this layer.
    '''

    def __init__(self, units, weights=None, bias=None, activation=None):
        self.neurons = [Neuron(weights, bias, activation)
                        for _ in range(units)]

    def forward(self, inputs):
        return [[neuron.process(inp) for neuron in self.neurons]
                for inp in inputs]
