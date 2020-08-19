"""Neural network models."""

from .exceptions import LayerBuildingError
from .layers import DenseLayer
from .layers import SoftmaxLayer
from . import activations


class Model:
    """Simple densely connected neural network model.

    Parameters
    ----------
    layers : list of :class:`enzo.layers.Layer`
        The list of layers in this model. The first layer in this list must have an
        explicit input length.
    """

    def __init__(self, layers):
        self.layers = []
        for layer in layers:
            self.add_layer(layer)

    def forward(self, samples):
        """Return and store in `self.outputs` the activation matrix of this layer
            after forward propagation.
        """
        self.layers[0].forward(samples)
        for layer, previous_layer in self._iter_layers_and_previous():
            layer.forward(previous_layer.outputs)
        return self.outputs

    @property
    def outputs(self):
        """list or ndarray : The activations of the final layer of this :class:`Model`
            for the most recent samples
        """
        return self.layers[-1].outputs

    def add_layer(self, layer):
        try:
            input_length = self.layers[-1].output_length
        except IndexError:  # adding first layer
            self._add_first_layer(layer)
        else:
            self._add_non_first_layer(layer, input_length=input_length)

    def _add_first_layer(self, layer):
        try:
            layer.build()
        except TypeError:
            raise LayerBuildingError(
                "first layer must be initialized with an input length"
            )
        self.layers.append(layer)

    def _add_non_first_layer(self, layer, input_length):
        if layer.activation is activations.softmax:
            layer.activation = activations.noactivation
            self.add_layer(layer)
            self.add_layer(SoftmaxLayer(layer.output_length))
        else:
            layer.build(input_length=input_length)
            self.layers.append(layer)

    def _iter_layers_and_previous(self):
        yield from zip(self.layers[1:], self.layers)
