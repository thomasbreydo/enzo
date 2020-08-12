"""Errors and exceptions"""


class BackPropagationBeforeForwardException(Exception):
    """Raise when back-propagation is run before forward-propagation"""


class LayerBuildingError(Exception):
    """Should be raised when :func:`enzo.layers.DenseLayer.build` fails."""
