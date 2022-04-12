import os

import numpy as np
import pytest

from dadapy import IdEstimation

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

X = np.load(filename)


def test_compute_id_2NN():
    """Test that the id estimations with the binomial estimator works correctly."""
    np.random.seed(0)

    ie = IdEstimation(coordinates=X)

    id_b = ie.compute_id_binomial_rk(0.2, 0.5)
    assert id_b == pytest.approx([2.08426, 0.33112, 0.150000], abs=1e-4, rel=1e-2)

    id_b = ie.compute_id_binomial_k(5, 0.5)
    assert id_b == pytest.approx([1.98391, 0.123781, 0.56159], abs=1e-4, rel=1e-2)
