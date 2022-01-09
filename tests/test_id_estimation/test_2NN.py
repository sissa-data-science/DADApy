import os

import numpy as np
import pytest

from dadapy import DensityEstimation


def test_compute_id_2NN():
    """Test that the id estimations with 2NN work correctly"""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    np.random.seed(0)

    X = np.load(filename)

    de = DensityEstimation(coordinates=X)

    de.compute_distances(maxk=25)

    de.compute_id_2NN()
    assert pytest.approx(de.intrinsic_dim, abs=0.01) == 1.85

    ### testing 2NN scaling
    ids, ids_err, rs = de.return_id_scaling_2NN()

    assert ids == pytest.approx([1.85, 1.77, 1.78, 2.31], abs=0.01)
    assert ids_err == pytest.approx([0.0, 0.11, 0.15, 0.19], abs=0.01)
    assert rs == pytest.approx([0.39, 0.58, 0.78, 1.13], abs=0.01)
