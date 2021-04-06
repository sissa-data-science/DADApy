import numpy as np
import pytest

from duly import Clustering


def test_clustering_basics():
    """Test the clustering operations work correctly"""
    X = np.random.uniform(size=(50, 2))

    cl = Clustering(coordinates=X)

    cl.compute_distances(maxk=25)

    cl.compute_id_2NN()
    assert pytest.approx(2.04, cl.id_selected)

    cl.compute_density_kNN(10)

    cl.compute_clustering()

    assert cl.Nclus_m == 1

