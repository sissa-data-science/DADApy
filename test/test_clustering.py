import os

import numpy as np
import pytest

from dadapy import Clustering


def test_clustering_basics():
    """Test the clustering operations work correctly"""
    filename = os.path.join(os.path.split(__file__)[0], "2gaussians_in_2d.npy")

    X = np.load(filename)

    cl = Clustering(coordinates=X)

    cl.compute_distances(maxk=25)

    cl.return_id_2NN()
    assert pytest.approx(1.85, cl.intrinsic_dim)

    cl.compute_density_kNN(10)

    cl.compute_clustering_pure_python()

    assert cl.N_clusters == 2

    expected_cluster_assignment = np.array(
        [
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
        ]
    )

    assert (cl.cluster_assignment == expected_cluster_assignment).all()
