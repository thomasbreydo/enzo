import pytest
from enzo import Neuron


def test_init():
    n = Neuron()
    assert isinstance(n, Neuron)
