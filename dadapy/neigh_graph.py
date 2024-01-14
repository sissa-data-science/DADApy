# Copyright 2021-2024 The DADApy Authors. All Rights Reserved.
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

"""
The *neighbourhood_graph* module contains the *NeighGraph* class.

It contains different methods and attributes which allow to exploit the structure of the directed neighbourhood graph.
"""

import multiprocessing
import time

import numpy as np
from scipy import sparse

from dadapy._cython import cython_grads as cgr
from dadapy.kstar import KStar

cores = multiprocessing.cpu_count()


class NeighGraph(KStar):
    """AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
       Can estimate the optimal number k* of neighbors for each points.

    Inherits from class KStar.
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    Can estimate the optimal number k* of neighbors for each points.
    Can compute the log-density and its error at each point choosing among various kNN-based methods.
    Can return an estimate of the gradient of the log-density at each point and an estimate of the error on
        each component.
    Can return an estimate of the linear deviation from constant density at each point and an estimate of the error on
        each component.

    Attributes:
        nspar (int): total number of edges in the directed graph defined by kstar (sum over all points of kstar minus N)
        nind_list (np.ndarray(int), optional): size nspar x 2. Each row is a couple of indices of the connected graph
            stored in order of increasing point index and increasing neighbour length (E.g.: in the first row (0,j), j
            is the nearest neighbour of the first point. In the second row (0,l), l is the second-nearest neighbour of
            the first point. In the last row (N-1,m) m is the kstar-1-th neighbour of the last point.)
            nind_iptr (np.array(int), optional): size N+1. For each elemen i stores the 0-th index in nind_list at which
            the edges starting from point i start. The last entry is set to nind_list.shape[0]
        common_neighs_array
        common_neighs_mat
        AAAAA (scipy.sparse.csr_matrix(float), optional): stored as a sparse symmetric matrix of size N x N. Entry (i,j)
            gives the common number of neighbours between points i and j. Such value is reliable only if j is in the
            neighbourhood of i or vice versa
        pearson
        neigh_vector_diffs (np.ndarray(float), optional): stores vector differences from each point to its k*-1 nearest
            neighbors. Accessed by the method return_vector_diffs(i,j) for each j in the neighbourhood of i
        neigh_dists (np.array(float), optional): stores distances from each point to its k*-1 nearest neighbors in the
            order defined by nind_list
    """

    def __init__(
        self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores
    ):
        """Initialise the DensityEstimation class."""
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            njobs=njobs,
        )

        self.nspar = None
        self.nind_list = None
        self.nind_iptr = None
        self.common_neighs_array = None
        self.common_neighs_mat = None
        self.pearson_array = None
        self.pearson_mat = None
        self.neigh_vector_diffs = None
        self.neigh_dists = None

    # ----------------------------------------------------------------------------------------------

    def compute_neigh_indices(self):
        """Computes the indices of all the couples [i,j] such that j is a neighbour of i up to the k*-th nearest
            (excluded).
        The couples of indices are stored in a numpy ndarray of rank 2 and secondary dimension = 2.
        The index of the corresponding AAAAAAAAAAAAA make indpointer which is a np.array of length N which indicates
            for each i the starting index of the corresponding [i,.] subarray.

        """

        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print("Computation of the neighbour indices started")

        sec = time.time()

        # self.get_vector_diffs = sparse.csr_matrix((self.N, self.N),dtype=np.int_)
        self.nind_list, self.nind_iptr = cgr.return_neigh_ind(
            self.dist_indices, self.kstar
        )

        self.nspar = len(self.nind_list)

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing neighbour indices".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_neigh_dists(self):
        """Computes the (directed) neighbour distances graph using kstar[i] neighbours for each point i.
        Distances are stored in np.array form according to the order of nind_list.

        """

        if self.distances is None:
            self.compute_distances()

        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print("Computation of the neighbour distances started")

        sec = time.time()

        self.neigh_dists = cgr.return_neigh_distances_array(
            self.distances, self.dist_indices, self.kstar
        )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing neighbour distances".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def return_sparse_distance_graph(self):
        """Returns the (directed) neighbour distances graph using kstar[i] neighbours for each point i in N x N sparse
        csr_matrix form."""

        if self.neigh_dists is None:
            self.compute_neigh_dists()

        if self.nind_list is None or self.nind_iptr is None:
            self.compute_neigh_indices()

        dgraph = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        for ind_spar, indices in enumerate(self.nind_list):
            dgraph[indices[0], indices[1]] = self.neigh_dists[ind_spar]

        return dgraph.tocsr()

    # ----------------------------------------------------------------------------------------------

    def compute_neigh_vector_diffs(self):
        """Compute the vector differences from each point to its k* nearest neighbors.
        The resulting vectors are stored in a numpy ndarray of rank 2 and secondary dimension = dims.
        The index of  scipy sparse csr_matrix format.

        AAA better implement periodicity to take both a scalar and a vector

        """
        # compute neighbour indices
        if self.nind_list is None:
            self.compute_neigh_indices()

        if self.verb:
            print("Computation of the vector differences started")
        sec = time.time()

        # self.get_vector_diffs = sparse.csr_matrix((self.N, self.N),dtype=np.int_)
        if self.period is None:
            self.neigh_vector_diffs = cgr.return_neigh_vector_diffs(
                self.X, self.nind_list
            )
        else:
            self.neigh_vector_diffs = cgr.return_neigh_vector_diffs_periodic(
                self.X, self.nind_list, self.period
            )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing vector differences".format(sec2 - sec))

        # ----------------------------------------------------------------------------------------------

        # def return_neigh_vector_diffs(self, i, j):
        """Return the vector difference between points i and j.

        Args:
            i and j, indices of the two points

        Returns:
            self.X[j] - self.X[i]
        """

    #    return self.neigh_vector_diffs[self.nind_mat[i, j]]

    # ----------------------------------------------------------------------------------------------

    def compute_common_neighs(self, comp_common_neighs_mat=False):
        """Compute the common number of neighbours between couple of points (i,j) such that j is\
        in the neighbourhod of i. The numbers are stored in a scipy sparse csr_matrix format.

        Args:

        Returns:

        """

        # compute neighbour indices
        if self.nind_list is None:
            self.compute_neigh_indices()

        if self.verb:
            print("Computation of the numbers of common neighbours started")

        sec = time.time()
        if comp_common_neighs_mat is True:
            (
                self.common_neighs_array,
                self.common_neighs_mat,
            ) = cgr.return_common_neighs_comp_mat(
                self.kstar, self.dist_indices, self.nind_list
            )
        else:
            self.common_neighs_array = cgr.return_common_neighs(
                self.kstar, self.dist_indices, self.nind_list
            )
        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds to carry out the computation.".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_pearson(self, comp_p_mat=False, method="jaccard"):
        """Compute the empiric
        common number of neighbours between couple of points (i,j) such that j is
        in the neighbourhod of i. The numbers are stored in a scipy sparse csr_matrix format.

        Args:
            chi_matrix (bool)
            method (): jaccard, geometric, squared_geometric


        Returns:

        """

        # check or compute common_neighs
        if self.common_neighs_array is None:
            self.compute_common_neighs()
        if self.verb:
            print("Estimation of the pearson correlation coefficient started")
        sec = time.time()
        k1 = self.kstar[self.nind_list[:, 0]]
        k2 = self.kstar[self.nind_list[:, 1]]
        # method to estimate pearson
        if method == "jaccard":
            self.pearson_array = (
                self.common_neighs_array * 1.0 / (k1 + k2 - self.common_neighs_array)
            )
        elif method == "geometric":
            self.pearson_array = self.common_neighs_array * 1.0 / np.sqrt(k1 * k2)
        elif method == "squared_geometric":
            self.pearson_array = (
                self.common_neighs_array * self.common_neighs_array * 1.0 / (k1 * k2)
            )
        else:
            raise ValueError("method not recognised")

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds to carry out the estimation.".format(sec2 - sec))

        # save in matrix form
        if comp_p_mat is True:
            p_mat = sparse.lil_matrix((self.N, self.N), dtype=np.float_)
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                p_mat[i, j] = self.pearson_array[nspar]
                if p_mat[j, i] == 0:
                    p_mat[j, i] = p_mat[i, j]
            self.pearson_mat = p_mat.todense()
            np.fill_diagonal(self.pearson_mat, 1.0)
