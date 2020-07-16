import pytest
import enzo.activation as act


def test_positive_relu():
    assert act.relu(484) == 484
    assert act.relu(0.0001) == 0.0001


def test_negative_relu():
    assert act.relu(-59) == 0
    assert act.relu(0) == 0
