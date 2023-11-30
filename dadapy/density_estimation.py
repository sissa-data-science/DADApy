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

"""
The *density_estimation* module contains the *DensityEstimation* class.

The different algorithms of density estimation are implemented as methods of this class.
"""

import multiprocessing
import time

import numpy as np
from scipy import sparse
from scipy import linalg as slin

from dadapy._cython import cython_density as cd
from dadapy._cython import cython_grads as cgr
from dadapy._utils.density_estimation import (
    return_not_normalised_density_kstarNN,
    return_not_normalised_density_PAk,
    return_not_normalised_density_PAk_optimized,
)
from dadapy._utils.utils import compute_cross_nn_distances
from dadapy.id_estimation import IdEstimation

cores = multiprocessing.cpu_count()


class DensityEstimation(IdEstimation):
    """Computes the log-density and its error at each point and other properties.

    Inherits from class IdEstimation. Can estimate the optimal number k* of neighbors for each points.
    Can compute the log-density and its error at each point choosing among various kNN-based methods.
    Can return an estimate of the gradient of the log-density at each point and an estimate of the error on each component.
    Can return an estimate of the linear deviation from constant density at each point and an estimate of the error on each component.

    Attributes:
        kstar (np.array(float)): array containing the chosen number k* in the neighbourhood of each of the N points
        dc (np.array(float), optional): array containing the distance of the k*th neighbor from each of the N points
        nspar (int): total number of edges in the directed graph defined by kstar (sum over all points of kstar minus N)
        nind_list (np.ndarray(int), optional): size nspar x 2. Each row is a couple of indices of the connected graph stored in order of increasing point index and increasing neighbour length (E.g.: in the first row (0,j), j is the nearest neighbour of the first point. In the second row (0,l), l is the second-nearest neighbour of the first point. In the last row (N-1,m) m is the kstar-1-th neighbour of the last point.)
        nind_iptr (np.array(int), optional): size N+1. For each elemen i stores the 0-th index in nind_list at which the edges starting from point i start. The last entry is set to nind_list.shape[0]
        common_neighs_array
        common_neighs_mat
        AAAAA (scipy.sparse.csr_matrix(float), optional): stored as a sparse symmetric matrix of size N x N. Entry (i,j) gives the common number of neighbours between points i and j. Such value is reliable only if j is in the neighbourhood of i or vice versa
        pearson
        neigh_vector_diffs (np.ndarray(float), optional): stores vector differences from each point to its k*-1 nearest neighbors. Accessed by the method return_vector_diffs(i,j) for each j in the neighbourhood of i
        neigh_dists (np.array(float), optional): stores distances from each point to its k*-1 nearest neighbors in the order defined by nind_list
        log_den (np.array(float), optional): array containing the N log-densities
        log_den_err (np.array(float), optional): array containing the N errors on the log_den
        grads (np.ndarray(float), optional): for each line i contains the gradient components estimated from from point i
        grads_var (np.ndarray(float), optional): for each line i contains the estimated variance of the gradient components at point i
        check_grads_covmat (bool, optional): it is flagged "True" when grads_var contains the variance-covariance matrices of the gradients
        Fij_array (list(np.array(float)), optional): stores for each couple in nind_list the estimates of deltaF_ij computed from point i as semisum of the gradients in i and minus the gradient in j
        Fij_var_array (np.array(float), optional): stores for each couple in nind_list the estimates of the squared errors on the values in Fij_array
        inv_deltaFs_cov

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

        self.kstar = None
        self.nspar = None
        self.nind_list = None
        self.nind_iptr = None
        self.common_neighs_array = None
        self.common_neighs_mat = None
        self.pearson_array = None
        self.pearson_mat = None
        self.neigh_vector_diffs = None
        self.neigh_dists = None
        self.dc = None
        self.log_den = None
        self.log_den_err = None
        self.grads = None
        self.grads_var = None
        self.check_grads_covmat = False
        self.Fij_array = None
        self.Fij_var_array = None

    # ----------------------------------------------------------------------------------------------

    def set_kstar(self, k=0):
        """Set all elements of kstar to a fixed value k.

        Reset all other class attributes (all depending on kstar).

        Args:
            k: number of neighbours used to compute the density it can be an iteger or an array of integers
        """
        if isinstance(k, np.ndarray):
            self.kstar = k
        else:
            self.kstar = np.full(self.N, k, dtype=int)

        self.nspar = None
        self.nind_list = None
        self.nind_iptr = None
        self.common_neighs_array = None
        self.common_neighs_mat = None
        self.pearson_array = None
        self.pearson_mat = None
        self.neigh_vector_diffs = None
        self.neigh_dists = None
        self.dc = None
        self.log_den = None
        self.log_den_err = None
        self.grads = None
        self.grads_var = None
        self.check_grads_covmat = False
        self.Fij_array = None
        self.Fij_var_array = None

    # ----------------------------------------------------------------------------------------------

    def compute_density_kNN(self, k=10, bias=False):
        """Compute the density of each point using a simple kNN estimator.

        Args:
            k (int): number of neighbours used to compute the density

        Returns:
            log_den (np.ndarray(float)): estimated log density
            log_den_err (np.ndarray(float)): estimated error on log density
        """
        if self.intrinsic_dim is None:
            _ = self.compute_id_2NN()

        if self.verb:
            print(f"k-NN density estimation started (k={k})")

        self.set_kstar(k)

        log_den, log_den_err, dc = return_not_normalised_density_kstarNN(
            self.distances,
            self.intrinsic_dim,
            self.kstar,
            interpolation=False,
            bias=bias,
        )

        # Normalise density
        log_den -= np.log(self.N)

        self.log_den = log_den
        self.log_den_err = log_den_err
        self.dc = dc

        if self.verb:
            print("k-NN density estimation finished")

        return self.log_den, self.log_den_err

    # ----------------------------------------------------------------------------------------------

    def compute_kstar(self, Dthr=23.92812698):
        """Compute an optimal choice of k for each point.

        Args:
            Dthr (float): Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
                to a p-value of 1e-6.

        """
        if self.intrinsic_dim is None:
            _ = self.compute_id_2NN()

        if self.verb:
            print(f"kstar estimation started, Dthr = {Dthr}")

        sec = time.time()

        kstar = cd._compute_kstar(
            self.intrinsic_dim,
            self.N,
            self.maxk,
            Dthr,
            self.dist_indices.astype("int64"),
            self.distances.astype("float64"),
        )

        self.set_kstar(kstar)

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing kstar".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_neigh_indices(self):
        """Computes the indices of all the couples [i,j] such that j is a neighbour of i up to the k*-th nearest (excluded).
        The couples of indices are stored in a numpy ndarray of rank 2 and secondary dimension = 2.
        The index of the corresponding AAAAAAAAAAAAA make indpointer which is a np.array of length N which indicates for each i the starting index of the corresponding [i,.] subarray.

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
        """Returns the (directed) neighbour distances graph using kstar[i] neighbours for each point i in N x N sparse csr_matrix form."""

        if self.neigh_dists is None:
            self.compute_neigh_dists()

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

    def compute_common_neighs(self,comp_common_neighs_mat=False):
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
            self.common_neighs_array, self.common_neighs_mat = cgr.return_common_neighs_comp_mat(
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

    def compute_pearson(self,comp_p_mat=False,method='jaccard'):
        """Compute the empiric 
        common number of neighbours between couple of points (i,j) such that j is\
        in the neighbourhod of i. The numbers are stored in a scipy sparse csr_matrix format.

        Args:
            chi_matrix (bool)
            method (): jaccard, geometric, squared_geometric


        Returns:

        """

        if self.pearson_array is None:
            # check or compute common_neighs
            if self.common_neighs_array is None:
                self.compute_common_neighs()
            if self.verb:
                print("Estimation of the pearson correlation coefficient started")
            sec = time.time()
            k1 = self.kstar[self.nind_list[:, 0]]
            k2 = self.kstar[self.nind_list[:, 1]]
            # method to estimate pearson
            if method=="jaccard":
                self.pearson_array = self.common_neighs_array*1. / (k1 + k2 - self.common_neighs_array)
            if method=="geometric":
                self.pearson_array = self.common_neighs_array*1. / np.sqrt(k1 * k2)
            if method=="squared_geometric":
                self.pearson_array = self.common_neighs_array*self.common_neighs_array*1. / (k1 * k2)
            sec2 = time.time()
            if self.verb:
                print("{0:0.2f} seconds to carry out the estimation.".format(sec2 - sec))

        # save in matrix form
        if comp_p_mat is True:
            if self.pearson_mat is None:
                p_mat = sparse.lil_matrix((self.N, self.N), dtype=np.float_)
                for nspar, indices in enumerate(self.nind_list):
                    i = indices[0]
                    j = indices[1]
                    p_mat[i, j] = self.pearson_array[nspar]
                    if p_mat[j,i] == 0:
                        p_mat[j,i] = p_mat[i,j]
                self.pearson_mat = p_mat.todense()
                np.fill_diagonal(self.pearson_mat, 1.)

        # AAAAAAAAAAAAA OPTIM: TESTARE SE FUNZIONA MEGLIO COL CICLO FOR O CON NUMPY NOTATION   
            

    # ----------------------------------------------------------------------------------------------

    def compute_density_kstarNN(self, Dthr=23.92812698, bias=False):
        """Compute the density of each point using a simple kNN estimator with an optimal choice of k.

        Args:
            Dthr (float): Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
                to a p-value of 1e-6.

        Returns:
            log_den (np.ndarray(float)): estimated log density
            log_den_err (np.ndarray(float)): estimated error on log density
        """
        self.compute_kstar(Dthr)

        if self.verb:
            print("kstar-NN density estimation started")

        log_den, log_den_err, dc = return_not_normalised_density_kstarNN(
            self.distances,
            self.intrinsic_dim,
            self.kstar,
            interpolation=False,
            bias=bias,
        )

        # Normalise density
        log_den -= np.log(self.N)

        self.log_den = log_den
        self.log_den_err = log_den_err
        self.dc = dc

        if self.verb:
            print("k-NN density estimation finished")

        return self.log_den, self.log_den_err

    # ----------------------------------------------------------------------------------------------

    def compute_density_kpeaks(self, Dthr=23.92812698):
        """Compute the density of each point as proportional to the optimal k value found for that point.

        This method is mostly useful for the kpeaks clustering algorithm.

        Args:
            Dthr: Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
                to a p-value of 1e-6.

        Returns:
            log_den (np.ndarray(float)): estimated log density
            log_den_err (np.ndarray(float)): estimated error on log density
        """
        self.compute_kstar(Dthr)

        if self.verb:
            print("Density estimation for k-peaks clustering started")

        dc = np.zeros(self.N, dtype=np.float_) 
        log_den = np.zeros(self.N, dtype=np.float_)
        log_den_err = np.zeros(self.N, dtype=np.float_)
        log_den_min = 9.9e300

        for i in range(self.N):
            k = self.kstar[i]
            dc[i] = self.distances[i, k]
            log_den[i] = k
            log_den_err[i] = 0
            for j in range(1, k):
                jj = self.dist_indices[i, j]
                log_den_err[i] = log_den_err[i] + (self.kstar[jj] - k) ** 2
            log_den_err[i] = np.sqrt(log_den_err[i] / k)

            if log_den[i] < log_den_min:
                log_den_min = log_den[i]

            # Normalise density

        self.log_den = log_den
        self.log_den_err = log_den_err
        self.dc = dc

        if self.verb:
            print("k-peaks density estimation finished")

        return self.log_den, self.log_den_err

    # ----------------------------------------------------------------------------------------------

    def compute_density_PAk(self, Dthr=23.92812698, optimized=True, bias=False):
        """Compute the density of each point using the PAk estimator.

        Args:
            Dthr (float): Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
                to a p-value of 1e-6.

        Returns:
            log_den (np.ndarray(float)): estimated log density
            log_den_err (np.ndarray(float)): estimated error on log density
        """
        # compute optimal k
        if self.kstar is None:
            self.compute_kstar(Dthr=Dthr)

        if self.verb:
            print("PAk density estimation started")

        sec = time.time()

        if optimized:
            log_den, log_den_err, dc = return_not_normalised_density_PAk_optimized(
                self.distances,
                self.intrinsic_dim,
                self.kstar,
                self.maxk,
                interpolation=False,
                bias=bias,
            )

        else:
            log_den, log_den_err, dc = return_not_normalised_density_PAk(
                self.distances,
                self.intrinsic_dim,
                self.kstar,
                self.maxk,
                interpolation=False,
                bias=bias,
            )

        sec2 = time.time()

        if self.verb:
            print(
                "{0:0.2f} seconds optimizing the likelihood for all the points".format(
                    sec2 - sec
                )
            )

        # Normalise density
        log_den -= np.log(self.N)

        self.log_den = log_den
        self.log_den_err = log_den_err
        self.dc = dc

        if self.verb:
            print("PAk density estimation finished")

        return self.log_den, self.log_den_err

    # ----------------------------------------------------------------------------------------------

    def compute_density_BMTI(self, use_variance=True, comp_err=False):
        # DENSITY_DEVELOP VERSION

        # compute changes in free energy
        if self.Fij_array is None:
            self.compute_deltaFs_grads_semisum()

        if self.verb:
            print("BMTI density estimation started")
            sec = time.time()

        # define the likelihood covarince matrix
        if use_variance:
            self.compute_deltaFs_inv_cross_covariance()

        sec2 = time.time()

        # compute adjacency matrix and cumulative changes
        A = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        supp_deltaF = sparse.lil_matrix((self.N, self.N), dtype=np.float_)


        if use_variance:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # tmp = 1.0 / self.Fij_var_array[nspar]
                tmp = self.inv_deltaFs_cov[nspar]
                A[i, j] = -tmp
                supp_deltaF[i, j] = self.Fij_array[nspar] * tmp
        else:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # A[i, j] = -1.0
                A[i, j] = -1.0
                supp_deltaF[i, j] = self.Fij_array[nspar]

        A = sparse.lil_matrix(A + A.transpose())

        diag = np.array(-A.sum(axis=1)).reshape((self.N,))
        # AAAAAAAAAAAAAAAA valutare se mettere 0.01*kstar

        A.setdiag(diag)

        # print("Diag = {}".format(diag))

        deltaFcum = np.array(supp_deltaF.sum(axis=0)).reshape((self.N,)) - np.array(
            supp_deltaF.sum(axis=1)
        ).reshape((self.N,))

        if self.verb:
            print("{0:0.2f} seconds to fill sparse matrix".format(time.time() - sec2))
        sec2 = time.time()

        log_den = sparse.linalg.spsolve(A.tocsr(), deltaFcum)

        if self.verb:
            print("{0:0.2f} seconds to solve linear system".format(time.time() - sec2))
        sec2 = time.time()

        self.log_den = log_den
        # self.log_den_err = np.sqrt((sparse.linalg.inv(A.tocsc())).diagonal())

        if comp_err is True:
            self.A = A.todense()
            self.B = slin.pinvh(self.A)
            # self.B = slin.inv(self.A)
            self.log_den_err = np.sqrt(np.diag(self.B))

            if self.verb:
                print("{0:0.2f} seconds inverting A matrix".format(time.time() - sec2))
            sec2 = time.time()

        # self.log_den_err = np.sqrt(np.diag(slin.pinvh(A.todense())))
        # self.log_den_err = np.sqrt(diag/np.array(np.sum(np.square(A.todense()),axis=1)).reshape(self.N,))

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds for BMTI density estimation".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_density_BMTI_wrong(self, use_variance=True, comp_err=True):
        # DENSITY_DEVELOP VERSION

        # compute changes in free energy
        if self.Fij_array is None:
            self.compute_deltaFs_grads_semisum()

        if self.verb:
            print("BMTI density estimation started")
            sec = time.time()

        if self.kstar is None:
            self.compute_kstar(Dthr=Dthr)

        # define the likelihood covarince matrix
        if use_variance:
            self.L_cov = np.zeros((self.N, self.N), dtype=np.float_)

            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # tmp = 1.0 / self.Fij_var_array[nspar]
                self.L_cov[i, j] = self.Fij_var_array[nspar]

            for i in range(self.N):
                self.L_cov[i,i] = 1./self.kstar[i]  #CONTROLLARE SE HA SENSO

            #invert it
            self.inv_L_cov = slin.pinvh(self.L_cov)

        if self.verb:
            print("{0:0.2f} seconds compute inverse covariance matrix for the likelihood".format(time.time()-sec))
        sec2 = time.time()


        # compute adjacency matrix and cumulative changes
        A = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        supp_deltaF = sparse.lil_matrix((self.N, self.N), dtype=np.float_)


        if use_variance:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # tmp = 1.0 / self.Fij_var_array[nspar]
                tmp = self.inv_L_cov[i,j]
                A[i, j] = -tmp
                supp_deltaF[i, j] = self.Fij_array[nspar] * tmp
        else:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # A[i, j] = -1.0
                A[i, j] = -1.0
                supp_deltaF[i, j] = self.Fij_array[nspar]

        A = sparse.lil_matrix(A + A.transpose())

        diag = np.array(-A.sum(axis=1)).reshape((self.N,))

        A.setdiag(diag)

        # print("Diag = {}".format(diag))

        deltaFcum = np.array(supp_deltaF.sum(axis=0)).reshape((self.N,)) - np.array(
            supp_deltaF.sum(axis=1)
        ).reshape((self.N,))

        if self.verb:
            print("{0:0.2f} seconds to fill sparse matrix".format(time.time() - sec2))
        sec2 = time.time()

        log_den = sparse.linalg.spsolve(A.tocsr(), deltaFcum)

        if self.verb:
            print("{0:0.2f} seconds to solve linear system".format(time.time() - sec2))
        sec2 = time.time()

        self.log_den = log_den
        # self.log_den_err = np.sqrt((sparse.linalg.inv(A.tocsc())).diagonal())

        if comp_err is True:
            self.A = A.todense()
            self.B = slin.pinvh(self.A)
            # self.B = slin.inv(self.A)
            self.log_den_err = np.sqrt(np.diag(self.B))

        if self.verb:
            print("{0:0.2f} seconds inverting A matrix".format(time.time() - sec2))
        sec2 = time.time()

        # self.log_den_err = np.sqrt(np.diag(slin.pinvh(A.todense())))
        # self.log_den_err = np.sqrt(diag/np.array(np.sum(np.square(A.todense()),axis=1)).reshape(self.N,))

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds for BMTI density estimation".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_density_gCorr_OLD(self, use_variance=True, comp_err=False):
        # TODO: matrix A should be in sparse format!

        # compute changes in free energy
        if self.Fij_array is None:
            self.compute_deltaFs_grads_semisum()

        if self.verb:
            print("gCorr density estimation started")
            sec = time.time()

        # compute adjacency matrix and cumulative changes
        A = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        supp_deltaF = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        # define redundancy factor for each A matrix entry as the geometric mean of the 2 corresponding k*
        k1 = self.kstar[self.nind_list[:, 0]]
        k2 = self.kstar[self.nind_list[:, 1]]

        #redundancy = np.sqrt(k1 * k2)
        redundancy = np.ones_like(k1,dtype=np.float_)
        # redundancy = k1
        # la cosa giusta e' forse sqrt(k1_inout * k2_inout) con ki_inout che conta tutte le volte che il punto i compare nella sommatoria
        #   cioe' ki + {numero di volte che k_i}
        # oppure (k1-1)/np.sqrt(k1*k2) che da' media ~1
        # redundancy = np.full_like(k1,fill_value=1.,dtype=np.float_)
        # print(redundancy)

        if use_variance:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # tmp = 1.0 / self.Fij_var_array[nspar]
                tmp = 1.0 / self.Fij_var_array[nspar] / redundancy[nspar]
                A[i, j] = -tmp
                supp_deltaF[i, j] = self.Fij_array[nspar] * tmp
        else:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # A[i, j] = -1.0
                A[i, j] = -1.0 / redundancy[nspar]
                supp_deltaF[i, j] = self.Fij_array[nspar]

        A = sparse.lil_matrix(A + A.transpose())

        diag = np.array(-A.sum(axis=1)).reshape((self.N,))

        A.setdiag(diag)

        # print("Diag = {}".format(diag))

        deltaFcum = np.array(supp_deltaF.sum(axis=0)).reshape((self.N,)) - np.array(
            supp_deltaF.sum(axis=1)
        ).reshape((self.N,))

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds to fill sparse matrix".format(sec2 - sec))

        log_den = sparse.linalg.spsolve(A.tocsr(), deltaFcum)

        if self.verb:
            print("{0:0.2f} seconds to solve linear system".format(time.time() - sec2))
        sec2 = time.time()

        self.log_den = log_den
        # self.log_den_err = np.sqrt((sparse.linalg.inv(A.tocsc())).diagonal())

        if comp_err is True:
            self.A = A.todense()
            self.B = slin.pinvh(self.A)
            # self.B = slin.inv(self.A)
            self.log_den_err = np.sqrt(np.diag(self.B))

        if self.verb:
            print("{0:0.2f} seconds inverting A matrix".format(time.time() - sec2))
        sec2 = time.time()

        # self.log_den_err = np.sqrt(np.diag(slin.pinvh(A.todense())))
        # self.log_den_err = np.sqrt(diag/np.array(np.sum(np.square(A.todense()),axis=1)).reshape(self.N,))

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds for gCorr density estimation".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_grads(self, comp_covmat=False):
        """Compute the gradient of the log density each point using k* nearest neighbors.
        The gradient is estimated via a linear expansion of the density propagated to the log-density.

        Args:

        Returns:

        MODIFICARE QUI E ANCHE NEGLI ATTRIBUTI

        """
        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        # check or compute vector_diffs
        if self.neigh_vector_diffs is None:
            self.compute_neigh_vector_diffs()

        if self.verb:
            print("Estimation of the density gradient started")

        sec = time.time()
        if comp_covmat is False:
            # self.grads, self.grads_var = cgr.return_grads_and_var_from_coords(self.X, self.dist_indices, self.kstar, self.intrinsic_dim)
            self.grads, self.grads_var = cgr.return_grads_and_var_from_nnvecdiffs(
                self.neigh_vector_diffs,
                self.nind_list,
                self.nind_iptr,
                self.kstar,
                self.intrinsic_dim,
            )
            self.grads_var = np.einsum(
                "ij, i -> ij", self.grads_var, self.kstar / (self.kstar - 1)
            )  # Bessel's correction for the unbiased sample variance estimator

        else:
            # self.grads, self.grads_var = cgr.return_grads_and_covmat_from_coords(self.X, self.dist_indices, self.kstar, self.intrinsic_dim)
            self.grads, self.grads_var = cgr.return_grads_and_covmat_from_nnvecdiffs(
                self.neigh_vector_diffs,
                self.nind_list,
                self.nind_iptr,
                self.kstar,
                self.intrinsic_dim,
            )
            self.check_grads_covmat = True

            self.grads_var = np.einsum(
                "ijk, i -> ijk", self.grads_var, self.kstar / (self.kstar - 1)
            )  # Bessel's correction for the unbiased sample variance estimator

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing gradients".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_deltaFs_grads_semisum(self, pearson_method="jaccard",comp_p_mat=False):
        """Compute deviations deltaFij to standard kNN log-densities at point j as seen from point i using\
            a linear expansion with as slope the semisum of the average gradient of the log-density over the neighbourhood of points i and j. \
            The parameter chi is used in the estimation of the squared error of the deltaFij as 1/4*(E_i^2+E_j^2+2*E_i*E_j*chi), \
            where E_i is the error on the estimate of grad_i*DeltaX_ij.

        Args:
            pearson_method: the Pearson correlation coefficient between the estimates of the gradient in i and j. Can take a numerical value between 0 and 1.\
                The option 'auto' takes a geometrical estimate of chi based on AAAAAAAAA

        Returns:

        """

        # check or compute vector_diffs
        if self.neigh_vector_diffs is None:
            self.compute_neigh_vector_diffs()

        # check or compute gradients and their covariance matrices
        if self.grads is None:
            self.compute_grads(comp_covmat=True)

        elif self.check_grads_covmat is False:
            self.compute_grads(comp_covmat=True)

        if self.verb:
            print(
                "Estimation of the gradient semisum (linear) corrections deltaFij to the log-density started"
            )
        sec = time.time()

        Fij_array = np.zeros(self.nspar)
        self.Fij_var_array = np.zeros(self.nspar)

        g1 = self.grads[self.nind_list[:, 0]]
        g2 = self.grads[self.nind_list[:, 1]]
        g_var1 = self.grads_var[self.nind_list[:, 0]]
        g_var2 = self.grads_var[self.nind_list[:, 1]]

        # check or compute common_neighs
        if self.pearson_mat is None:
            self.compute_pearson(method=pearson_method,comp_p_mat=comp_p_mat)

  
        Fij_array = 0.5 * np.einsum("ij, ij -> i", g1 + g2, self.neigh_vector_diffs)
        vari = np.einsum(
            "ij, ij -> i",
            self.neigh_vector_diffs,
            np.einsum("ijk, ik -> ij", g_var1, self.neigh_vector_diffs),
        )
        varj = np.einsum(
            "ij, ij -> i",
            self.neigh_vector_diffs,
            np.einsum("ijk, ik -> ij", g_var2, self.neigh_vector_diffs),
        )
        self.Fij_var_array = 0.25 * (vari + varj + 2 * self.pearson_array * np.sqrt(vari * varj))

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing gradient corrections".format(sec2 - sec))

        self.Fij_array = Fij_array
        self.Fij_var_array = self.Fij_var_array
        # self.Fij_var_array = self.Fij_var_array*k1/(k1-1) #Bessel's correction for the unbiased sample variance estimator

    # ----------------------------------------------------------------------------------------------

    def compute_deltaFs_inv_cross_covariance(self,pearson_method="jaccard"):
        """Compute the cross-covariance of the deltaFs cov[deltaFij,deltaFlm] using cython.

        Args: AAAAAAAAAAAAAAAAA

        Returns: AAAAAAAAAAAAAAAAA

        """

        # check for deltaFs
        if self.pearson_mat is None:
            self.compute_pearson(method=pearson_method,comp_p_mat=True)

        # check or compute deltaFs_grads_semisum
        if self.Fij_var_array is None:
            self.compute_deltaFs_grads_semisum()
        # AAAAAAAAAAAAAAA controllare se serve
        #smallnumber = 1.e-10
        #data.grads_var += smallnumber*np.tile(np.eye(data.dims),(data.N,1,1))
        # AAAAAAAAAAAAAAA fine controllare se serve

        if self.verb:
            print(
                "Estimation of the deltaFs cross-covariance started"
            )
        sec = time.time()
        self.inv_deltaFs_cov = cgr.return_deltaFs_inv_cross_covariance(
                        self.grads_var,
                        self.neigh_vector_diffs,
                        self.nind_list,
                        self.pearson_mat,
                        self.Fij_var_array,
        )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing the deltaFs cross-covariance".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def return_entropy(self):
        """Compute a very rough estimate of the entropy of the data distribution.

        The cimputation simply returns the average negative log probability estimates.

        Returns:
            H (float): the estimate entropy of the distribution

        """
        assert self.log_den is not None

        H = -np.mean(self.log_den)

        return H

    # ----------------------------------------------------------------------------------------------

    def return_interpolated_density_kNN(self, X_new, k):
        """Return the kNN density of the primary dataset, evaluated on a new set of points "X_new".

        Args:
            X_new (np.ndarray(float)): The points onto which the density should be computed
            k (int): the number of neighbours considered for the kNN estimator

        Returns:
            log_den (np.ndarray(float)): log density of dataset evaluated on X_new
            log_den_err (np.ndarray(float)): error on log density estimates
        """
        assert self.X is not None

        if self.intrinsic_dim is None:
            _ = self.compute_id_2NN()

        cross_distances, cross_dist_indices = compute_cross_nn_distances(
            X_new, self.X, self.maxk, self.metric, self.period
        )

        kstar = np.ones(X_new.shape[0], dtype=int) * k

        log_den, log_den_err, dc = return_not_normalised_density_kstarNN(
            cross_distances, self.intrinsic_dim, kstar, interpolation=True
        )

        # Normalise density
        log_den -= np.log(self.N)

        return log_den, log_den_err

    # ----------------------------------------------------------------------------------------------

    def return_interpolated_density_kstarNN(self, X_new, Dthr=23.92812698):
        """Return the kstarNN density of the primary dataset, evaluated on a new set of points "X_new".

        Args:
            X_new (np.ndarray(float)): The points onto which the density should be computed
            Dthr: Likelihood ratio parameter used to compute optimal k

        Returns:
            log_den (np.ndarray(float)): log density of dataset evaluated on X_new
            log_den_err (np.ndarray(float)): error on log density estimates
        """
        assert self.X is not None

        if self.intrinsic_dim is None:
            _ = self.compute_id_2NN()

        cross_distances, cross_dist_indices = compute_cross_nn_distances(
            X_new, self.X, self.maxk, self.metric, self.period
        )

        kstar = cd._compute_kstar_interp(
            self.intrinsic_dim,
            X_new.shape[0],
            self.maxk,
            Dthr,
            cross_dist_indices,
            cross_distances,
            self.distances,
        )

        log_den, log_den_err, dc = return_not_normalised_density_kstarNN(
            cross_distances, self.intrinsic_dim, kstar, interpolation=True
        )

        # Normalise density
        log_den -= np.log(self.N)

        return log_den, log_den_err

    # ----------------------------------------------------------------------------------------------

    def return_interpolated_density_PAk(self, X_new, Dthr=23.92812698):
        """Return the PAk density of the primary dataset, evaluated on a new set of points "X_new".

        Args:
            X_new (np.ndarray(float)): The points onto which the density should be computed
            Dthr: Likelihood ratio parameter used to compute optimal k

        Returns:
            log_den (np.ndarray(float)): log density of dataset evaluated on X_new
            log_den_err (np.ndarray(float)): error on log density estimates
        """
        assert self.X is not None

        if self.intrinsic_dim is None:
            _ = self.compute_id_2NN()

        cross_distances, cross_dist_indices = compute_cross_nn_distances(
            X_new, self.X, self.maxk, self.metric, self.period
        )

        kstar = cd._compute_kstar_interp(
            self.intrinsic_dim,
            X_new.shape[0],
            self.maxk,
            Dthr,
            cross_dist_indices,
            cross_distances,
            self.distances,
        )

        log_den, log_den_err, dc = return_not_normalised_density_PAk(
            cross_distances, self.intrinsic_dim, kstar, self.maxk, interpolation=True
        )

        # Normalise density
        log_den -= np.log(self.N)

        return log_den, log_den_err
