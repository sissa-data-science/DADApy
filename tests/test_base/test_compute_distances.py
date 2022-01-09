import os

import numpy as np
import pytest

from dadapy import Base


def test_compute_distances():
    """Test the computation of distances"""

    X = np.array([[0, 0, 0], [0.5, 0, 0], [0.9, 0, 0]])

    base = Base(coordinates=X)

    base.compute_distances()

    expected_dists = [[0.0, 0.5, 0.9], [0.0, 0.4, 0.5], [0.0, 0.4, 0.9]]

    expected_ind = [[0, 1, 2], [1, 2, 0], [2, 1, 0]]

    assert pytest.approx(base.distances) == expected_dists
    assert pytest.approx(base.dist_indices) == expected_ind

    base.compute_distances(period=1.1, metric="manhattan")

    expected_dists = [[0.0, 0.2, 0.5], [0.0, 0.4, 0.5], [0.0, 0.2, 0.4]]

    expected_ind = [[0, 2, 1], [1, 2, 0], [2, 0, 1]]

    assert pytest.approx(base.distances) == expected_dists
    assert pytest.approx(base.dist_indices) == expected_ind
