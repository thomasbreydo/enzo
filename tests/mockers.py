import pytest
import numpy as np

MOCKED_RANDOM_WEIGHTS = {}


@pytest.fixture
def mock_np_random_rand(monkeypatch):
    def new_rand(*args):
        return MOCKED_RANDOM_WEIGHTS[args]

    monkeypatch.setattr(np.random, "rand", new_rand)
