"""Errors and exceptions"""


class BackPropagationBeforeForwardException(Exception):
    """Raise when back-propagation is run before forward-propagation"""


red


class LayerBuildingError(Exception):
    """Raise when :func:`layers.Layer.build` fails."""
