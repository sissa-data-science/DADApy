import os

import numpy as np
import pytest

from duly import Clustering


def test_clustering_basics():
    """Test the clustering operations work correctly"""
    filename = os.path.join(os.path.split(__file__)[0], "2gaussians_in_2d.npy")

    X = np.load(filename)

    cl = Clustering(coordinates=X)

    cl.compute_distances(maxk=25)

    cl.compute_id_2NN()
    assert pytest.approx(1.85, cl.id_selected)

    cl.compute_density_kNN(10)

    cl.compute_clustering()

    assert cl.Nclus_m == 2

    expected_labels = np.array(
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

    assert (cl.labels == expected_labels).all()
