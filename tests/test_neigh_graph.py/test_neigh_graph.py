# Copyright 2021-2023 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for testing the NeighGraph class."""

import numpy as np
from dadapy import NeighGraph


# define a basic dataset with 6 points
data = np.array([[0, 0], [0.15, 0], [0.2, 0], [4, 0], [4.1, 0], [4.2, 0]])

# list of neighbor pairs
expected_nint_list = [[0, 1],
 [1, 2],
 [2, 1],
 [3, 4],
 [4, 3],
 [5, 4]]

# list of expected indices of the neighbor pairs TODO: Matteo, is this correct?
expected_nind_iprt = [0, 1, 2, 3, 4, 5, 6]

# number of neighbour pairs
expected_nspar = 6

expected_neigh_dists = np.array([0.15, 0.05, 0.05, 0.1, 0.1,  0.1])

def test_compute_neigh_indices():
    """Test the compute_neigh_indices method."""
    # create the NeighGraph object
    neigh_graph = NeighGraph(coordinates=data)
    neigh_graph.compute_kstar(Dthr=0.0)
    # compute the indices of the neighbors
    neigh_graph.compute_neigh_indices()
    # check that the result is correct
    assert np.array_equal(neigh_graph.nind_list, expected_nint_list)
    assert np.array_equal(neigh_graph.nind_iptr, expected_nind_iprt)
    assert neigh_graph.nspar == expected_nspar


def test_compute_neigh_dists():
    """Test the compute_neigh_dists method."""
    # create the NeighGraph object
    neigh_graph = NeighGraph(coordinates=data)
    neigh_graph.compute_kstar(Dthr=0.0)
    # compute the distances of the neighbors
    neigh_graph.compute_neigh_dists()
    # check that the result is correct
    assert np.allclose(neigh_graph.neigh_dists, expected_neigh_dists)


def test_return_sparse_distance_graph():
    """Test the return_sparse_distance_graph method."""
    # create the NeighGraph object
    neigh_graph = NeighGraph(coordinates=data)
    neigh_graph.compute_kstar(Dthr=0.0)
    graph = neigh_graph.return_sparse_distance_graph()
    # check that the result is correct
    print(graph)
    # assert np.array_equal(neigh_graph.nind_list, expected_nint_list)
    # assert np.array_equal(neigh_graph.nind_iptr, expected_nind_iprt)
    # assert neigh_graph.nspar == expected_nspar
    # assert np.allclose(neigh_graph.neigh_dists, expected_neigh_dists)

print(test_return_sparse_distance_graph())
