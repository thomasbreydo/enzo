"""Neural network models"""

from .exceptions import LayerBuildingError


class Model:
    """Simple densely connected neural network model.
    
    Parameters
    ----------
    layers : list of :class:`enzo.layers.Layer`
        The list of layers in this model. The first layer in this list must have an
        explicit input length.
    """

    def __init__(self, layers):
        self.layers = layers
        self._build_layers()

    def _build_layers(self):
        self._build_first_layer()
        for layer, previous_layer in self._iter_layers_and_previous():
            layer.build(input_length=previous_layer.output_length)

    def _build_first_layer(self):
        try:
            self.layers[0].build()
        except TypeError:
            raise LayerBuildingError(
                "first layer must be initialized with an input length"
            )

    def _iter_layers_and_previous(self):
        yield from zip(self.layers[1:], self.layers)

    def forward(self, samples):
        self.layers[0].forward(samples)
        for layer, previous_layer in self._iter_layers_and_previous():
            layer.forward(previous_layer.outputs)

    @property
    def outputs(self):
        return self.layers[-1].outputs
