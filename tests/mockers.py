import pytest
import numpy as np
import copy


class SetOnceDict:
    def __init__(self):
        self._dict = {}

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if key in self._dict:
            raise ValueError(f"key {key!r} already exists.")
        self._dict.__setitem__(key, value)


MOCKED_RANDOM_WEIGHTS = SetOnceDict()


@pytest.fixture
def mock_np_random_randn(monkeypatch):
    def new_randn(*args):
        return copy.deepcopy(MOCKED_RANDOM_WEIGHTS[args])

    monkeypatch.setattr(np.random, "randn", new_randn)
