import os

import numpy as np

from dadapy import DensityEstimation

expected_kstar = [
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    16,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
    23,
]


def test_compute_kstar():
    """Test that compute_kstar works correctly"""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    X = np.load(filename)[:25]

    de = DensityEstimation(coordinates=X)

    de.compute_kstar()

    assert (de.kstar == expected_kstar).all()
