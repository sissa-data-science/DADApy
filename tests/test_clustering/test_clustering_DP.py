import os

import numpy as np

from dadapy import Clustering

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

X = np.load(filename)

expected_cluster_assignment = np.array(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
)


def test_compute_DecGraph():
    """Test the compute_DecGraph function works correctly"""

    cl = Clustering(coordinates=X)

    cl.compute_density_PAk()

    cl.compute_DecGraph()

    assert np.count_nonzero(cl.delta > 1) == 7


def test_compute_cluster_DP():
    """Test the DP clustering works correctly"""

    cl = Clustering(coordinates=X)
    cl.compute_density_PAk()
    cl.compute_DecGraph()
    cl.compute_cluster_DP(dens_cut=-3.0, delta_cut=3.0)
    assert (cl.cluster_assignment == expected_cluster_assignment).all()
