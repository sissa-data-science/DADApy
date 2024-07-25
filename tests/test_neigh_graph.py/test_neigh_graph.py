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
data = np.array([[0, 0], [0.15, 0], [0.2, 0], [4, 0], [4.09, 0], [4.2, 0]])

# list of neighbor pairs
expected_nint_list = [[0, 1], [1, 2], [2, 1], [3, 4], [4, 3], [5, 4]]

# list of expected indices of the neighbor pairs TODO: Matteo, is this correct?
expected_nind_iprt = [0, 1, 2, 3, 4, 5, 6]

# number of neighbour pairs
expected_nspar = 6

expected_neigh_dists = np.array([0.15, 0.05, 0.05, 0.09, 0.09, 0.11])

expected_distance_graph = [
    [0.0, 0.15, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
    [0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.09, 0.0],
    [0.0, 0.0, 0.0, 0.09, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.11, 0.0],
]

neigh_vector_diffs = [
    [0.15, 0.0],
    [0.05, 0.0],
    [-0.05, 0.0],
    [0.09, 0.0],
    [-0.09, 0.0],
    [-0.11, 0.0],
]


expected_common_neighs_array = [1, 2, 2, 2, 2, 1]
expected_common_neighs_mat = [
    [0, 1, 0, 0, 0, 0],
    [1, 0, 2, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0],
    [0, 0, 0, 2, 0, 1],
    [0, 0, 0, 0, 1, 0],
]


expected_neigh_similarity_index = np.array([1.0 / 3.0, 1.0, 1.0, 1.0, 1.0, 1.0 / 3.0])

expected_neigh_similarity_index_mat = np.array(
    [
        [1.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 3.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0 / 3.0],
        [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 1.0],
    ]
)


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
    assert np.allclose(graph.toarray(), expected_distance_graph)


def test_compute_neigh_vector_diffs():
    """Test the compute_neigh_vector_diffs method."""
    # create the NeighGraph object
    neigh_graph = NeighGraph(coordinates=data)
    neigh_graph.compute_kstar(Dthr=0.0)
    # compute the distances of the neighbors
    neigh_graph.compute_neigh_vector_diffs()
    # check that the result is correct
    assert np.allclose(neigh_graph.neigh_vector_diffs, neigh_vector_diffs)


def test_compute_common_neighs():
    """Test the compute_common_neighs method."""
    # create the NeighGraph object
    neigh_graph = NeighGraph(coordinates=data)
    neigh_graph.compute_distances()
    neigh_graph.set_kstar([2, 2, 2, 2, 2, 2])

    neigh_graph.compute_common_neighs(comp_common_neighs_mat=False)
    print(neigh_graph.common_neighs_array)
    assert np.array_equal(neigh_graph.common_neighs_array, expected_common_neighs_array)

    neigh_graph.compute_common_neighs(comp_common_neighs_mat=True)
    print(neigh_graph.common_neighs_mat)
    assert np.array_equal(neigh_graph.common_neighs_mat, expected_common_neighs_mat)


def test_compute_neigh_similarity_index():
    """Test the compute_neigh_similarity_index."""
    # create the NeighGraph object
    neigh_graph = NeighGraph(coordinates=data)
    neigh_graph.compute_distances()
    neigh_graph.set_kstar([2, 2, 2, 2, 2, 2])
    neigh_graph.compute_neigh_similarity_index()
    assert np.allclose(
        neigh_graph.neigh_similarity_index, expected_neigh_similarity_index
    )

def test_compute_neigh_similarity_index_mat():
    """Test the compute_neigh_similarity_index_mat."""
    # create the NeighGraph object
    neigh_graph = NeighGraph(coordinates=data)
    neigh_graph.compute_distances()
    neigh_graph.set_kstar([2, 2, 2, 2, 2, 2])
    neigh_graph.compute_neigh_similarity_index_mat()
    assert np.allclose(
        neigh_graph.neigh_similarity_index_mat, expected_neigh_similarity_index_mat
    )