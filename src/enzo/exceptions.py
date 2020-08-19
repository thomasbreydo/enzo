"""Errors and exceptions."""


class BackBeforeForwardException(Exception):
    """Raise when back-propagation is run before forward-propagation."""


class LayerBuildingError(Exception):
    """Raise when :func:`enzo.layers.DenseLayer.build` fails."""
