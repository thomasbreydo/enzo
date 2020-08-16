import pytest
import enzo.activations as act
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


def test_softmax():
    after_softmax = act.softmax([[1, 2, 3, 4, 5], [-10, 4, 0, 2, 2]])
    np.testing.assert_allclose(
        after_softmax,
        np.array(
            [
                [0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865],
                [
                    6.4510246829898e-07,
                    0.77580299210163,
                    0.014209327452133,
                    0.10499351767189,
                    0.10499351767189,
                ],
            ],
        ),
    )
