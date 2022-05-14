from dadapy._utils import utils
import numpy as np
import pytest


def test_zero_dist():
    X = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    with pytest.warns(UserWarning):
        utils.compute_nn_distances(X, maxk=2, metric="euclidean", period=None)
