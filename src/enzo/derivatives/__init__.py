"""Derivative functions for functions."""

from .wrappers import with_derivative
from .d_activations import d_noactivation
from .d_activations import d_relu
from .d_activations import d_sigmoid
from .d_activations import d_softmax
from .d_losses import d_crossentropy
