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
    """
    Computes the directed neighbourhood graph (DNG) based on the kstar optimal neighbourhood selection and
    other DNG-based quantities. Inherits from class KStar. The DNG is stored in nind_list and can be
    retrieved using the kstar (inherited from Kstar class) and nind_iptr attributes. Can compute and store
    distances and vector differences between nodes connected on the DNG. Can compute and store the number
    of points in common in the neighbourhoods of couples of nodes connected on the DNG. Can use the common
    neighbours to give a geometric estimate of the overlap between neighbourhoods (various methods
    implemented).

    Attributes:
        nspar (int): total number of edges in the (sparse) directed graph defined by kstar i.e. the sum
            over all points of (kstar -1).
        nind_list (np.ndarray(int), optional): size nspar x 2. Each row contains a couple of indices of
            edges connected in the DNG stored in order of increasing point index and increasing neighbour
            rank. Therefore, nind_list assigns a unique index from 0 to nspar-1 to all the edges in the DNG.
        nind_iptr (np.array(int), optional): size N+1. For each element i, it stores the 0-th index in
            nind_list at which all the edges of the form [i,.] (i.e. the ones connecting point i to its
            neighbours) start. The last entry is set to nind_list.shape[0].
        common_neighs_array (np.array(int), optional): size nspar. At position  p, it contains the total
            number of points in common between the neighbourhoods of the two points forming the p-th
            directed edge of the neighbourhood graph.
        common_neighs_mat (np.ndarray(float), optional): size N x N. Entry (i,j) contains the total number
            of points in common between the neighbourhoods of points i and j.
        pearson_array (np.ndarray(float), optional): size nspar. At position p, it contains an estimate of
            the overlap between the neighbourhoods of the two points forming the p-th directed edge of the
            neighbourhood graph.
        pearson_mat (np.ndarray(float), optional): size N x N. Entry (i,j) contains an estimate of the
            overlap between the neighbourhoods of the two points i and j.
        neigh_vector_diffs (np.ndarray(float), optional): size nspar x dims. At position p, it stores the
            vector difference from point nind_list[p,0] to point nind_list[p,1].
        neigh_dists (np.array(float), optional): size nspar. Stores the distances from each point to its
            k*-1 nearest neighbors in the order defined by nind_list.
    """

    def __init__(
        self, coordinates=None, distances=None, maxk=None, verbose=False, n_jobs=cores
    ):
        """Initialise the DensityEstimation class."""
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            n_jobs=n_jobs,
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

    def set_kstar(self, k=0):
        """Set all elements of kstar to a fixed value k.

        Overload set_kstar of the superior class (Kstar).
        First, call the set_kstar from the KStar class.
        Then also reset all other NeighGraph attributes depending on kstar to None.

        Args:
            k: number of neighbours used to compute the density. It can be an iteger or an array of integers
        """
        super().set_kstar(k)

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
        """
        Compute indices of all couples [i,j] where j is a neighbour of i up to k*-th nearest (excluded).
        The couples of indices are stored in nind_list.
        Also compute and fill the attributes nspar (the 0-th shape of nind_list) nind_iptr.
        """

        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print("Computation of the neighbour indices started")

        sec = time.time()

        self.nind_list, self.nind_iptr = cgr.return_neigh_ind(
            self.dist_indices, self.kstar
        )

        self.nspar = len(self.nind_list)

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing neighbour indices".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_neigh_dists(self):
        """Compute the (directed) neighbour distances graph using kstar[i] neighbours for each point i.

        Distances are stored in the neigh_dists array.

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
        """Return the (directed) neighbour distances graph as a N x N scipy sparse csr_matrix form.

        If the attribute neigh_dists is not assigned, invokes method compute_neigh_dists.

        """

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
        """Compute the vector differences from each point to its kstar nearest neighbors.

        The resulting vectors are stored in neigh_vector_diffs.

        """
        # compute neighbour indices
        if self.nind_list is None:
            self.compute_neigh_indices()

        if self.verb:
            print("Computation of the vector differences started")
        sec = time.time()

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

    def compute_common_neighs(self, comp_common_neighs_mat=False):
        """Compute the common number of neighbours between the couple of points (i,j) such that j is
        in the neighbourhod of i.

        The numbers are stored in common_neighs_array.
        If the flag comp_common_neighs_mat has value True, also the symmetric matrix common_neighs_mat is computed.

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
        """
        Compute an estimate of the overlaps between the neighbourhoods of the points connected by edges on the DNG
        values from 0 to 1 and stores them in the pearson_array attribute. See also the documentation for the
        pearson_array attribute for completeness.

        Args:
            comp_p_mat (bool): if True, also computes the pearson_mat attribute.
            method (str): currently implemented "jaccard", "geometric", "squared_geometric".
            Let us denote the neighbourhoods of points 1 and 2 respectively by the sets Ω_1 and Ω_2.
            Then k_1 = #Ω_1 and k_2 = #Ω_2 are the neighbourhood sizes and k_1,2 = Ω_1 ∩ Ω_2 the number of points
            in common between the two neighbourhoods (which can be read off at common_neighs_mat[1,2]
            if common_neighs_mat has been computed). The methods to compute the Pearson coefficients are:

            "jaccard": p_1,2 = k_1,2 / (k_1 + k_2 - k_1,2) = #(Ω_1 ∩ Ω_2) / #(Ω_1 ∪ Ω_2), i.e. the Jaccard index
            "geometric": p_1,2 = k_1,2 / sqrt(k_1 * k_2), i.e. the number of common points divided by the geometric mean
            "squared geometric": p_1,2 = (k_1,2)^2 / (k_1 * k_2), i.e. the square of the "geometric" version
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
