import os

import numpy as np

from dadapy import Clustering

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

X = np.load(filename)

# TODO: @alexdepremia you are surely the best person to fill this in
#  note that the dataset X is a very simple dataset with 2 well separated Gaussians
def test_compute_DecGraph():
    """Test the compute_DecGraph function works correctly"""

    # cl = Clustering(coordinates=X)

    # cl.compute_DecGraph()

    # ... some assertation

    assert True


# TODO: @alexdepremia you are surely the best person to fill this in
#  note that the dataset X is a very simple dataset with 2 well separated Gaussians
def test_compute_cluster_DP():
    """Test the DP clustering works correctly"""

    # cl = Clustering(coordinates=X)

    # cl.compute_cluster_DP()

    # ... some assertation

    assert True
