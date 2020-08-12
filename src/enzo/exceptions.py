"""Errors and exceptions"""


class BackPropagationBeforeForwardException(Exception):
    """Raise when back-propagation is run before forward-propagation"""

    pass


class LayerBuildingError(Exception):
    """Should be raised when :func:`layers.Layer.build` fails."""
