import numpy as np
import pytest

from dadapy._utils import utils


def test_zero_dist():
    X = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    with pytest.warns(UserWarning):
        utils.compute_nn_distances(X, maxk=2, metric="euclidean", period=None)
