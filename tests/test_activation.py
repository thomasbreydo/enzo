import pytest
import enzo.activation as act
import numpy as np


def test_positive_relu():
    assert act.relu(484) == 484
    assert act.relu(0.0001) == 0.0001


def test_negative_relu():
    assert act.relu(-59) == 0
    assert act.relu(0) == 0


def test_positive_sigmoid():
    assert act.sigmoid(5) == pytest.approx(0.9933071490757153)
    assert act.sigmoid(5000) == pytest.approx(1)


def test_negative_sigmoid():
    assert act.sigmoid(-9) == pytest.approx(0.00012339457598623172)
    assert act.sigmoid(-5000) == pytest.approx(0)


def test_zero_sigmoid():
    assert act.sigmoid(0) == pytest.approx(0.5)


def test_noactivation():
    random_arr = np.random.randn(5, 9)
    assert (act.noactivation(random_arr) == random_arr).all()
