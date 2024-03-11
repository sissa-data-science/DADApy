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
The *density_advanced* module contains the *DensityEstimation* class.

Different algorithms to estimate the logdensity, the logdensity gradientest and the logdensity differences are
implemented as methods of this class. In particular, differently from the methods implemented in the DensityEstimation,
the methods in the DensityAdvanced class are based on the sparse neighbourhood graph structure which is implemented
in the NeighGraph class.
"""

import multiprocessing
import time
import warnings

import numpy as np
from scipy import linalg as slin
from scipy import sparse

from dadapy._cython import cython_grads as cgr
from dadapy._utils.density_estimation import return_not_normalised_density_kstarNN
from dadapy.density_estimation import DensityEstimation
from dadapy.neigh_graph import NeighGraph

cores = multiprocessing.cpu_count()


class DensityAdvanced(DensityEstimation, NeighGraph):
    """Computes the log-density gradient and its covariance at each point and other log-density-related properties.

    Can return an estimate of the gradient of the log-density at each point and an estimate of the error on each
    component.
    Can return an estimate of log-density differences and their error each point based on the gradient estimates.
    Can compute the log-density and its error at each point using BMTI, i.e. integrating the log-density differences
    on the neighbourhood graph

    Attributes:
        grads (np.ndarray(float), optional): size N. Contains the gradient components estimated at each point i
        grads_var (np.ndarray(float), optional): size N x dims. For each line i contains the estimated variance of the
            gradient components at point i
        grads_covmat (np.ndarray(float), optional): size N x dims x dims. For each line i contains the estimated
            covariance matrix of the gradient components at point i
        Fij_array (list(np.array(float)), optional): size nspar. Stores for each couple in nind_list the estimates of
            deltaF_ij computed from point i as semisum of the gradients in i and minus the gradient in j
        Fij_var_array (np.array(float), optional): size nspar. Stores for each couple in nind_list the estimates of the
            squared errors on the values in Fij_array
        inv_deltaFs_cov (np.array(float), optional): size nspar. Stores for each couple in nind_list the estimates of
            the inverse cross-covariance of the deltaFs, that is: cov [ deltaFij , deltaFlm ] .

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

        self.grads = None
        self.grads_var = None
        self.grads_covmat = None
        self.Fij_array = None
        self.Fij_var_array = None
        self.inv_deltaFs_cov = None

    # ----------------------------------------------------------------------------------------------

    def set_kstar(self, k=0):
        """Set all elements of kstar to a fixed value k.

        Overload the set_kstar method from the superior classes.
        First, call the set_kstar from the superior classes.
        Then also reset all other AdvanceDensity attributes depending on kstar to None.

        Args:
            k: number of neighbours used to compute the density. It can be an iteger or an array of integers
        """
        super().set_kstar(k)

        self.grads = None
        self.grads_var = None
        self.grads_covmat = None
        self.Fij_array = None
        self.Fij_var_array = None
        self.inv_deltaFs_cov = None

    # ----------------------------------------------------------------------------------------------

    def compute_grads(self, comp_covmat=False):
        """Compute the gradient of the log density each point using kstar nearest neighbors and store

        Estimate the gradient using an improved version of the mean-shift gradient algorithm [Fukunaga1975] as
        presented in [Carli2024].
        Store the computed gradients in grads.
        Also compute the variance of the gradient and store it in grads_var.
        Optionally, the whole covariance matrix can be estimated for gradient

        Args:
            comp_covmat (bool): if True, the whole covariance matrix is computed for each gradient and stored in
            grads_covmat

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
            self.grads, self.grads_covmat = cgr.return_grads_and_covmat_from_nnvecdiffs(
                self.neigh_vector_diffs,
                self.nind_list,
                self.nind_iptr,
                self.kstar,
                self.intrinsic_dim,
            )

            # Bessel's correction for the unbiased sample variance estimator
            self.grads_covmat = np.einsum(
                "ijk, i -> ijk", self.grads_covmat, self.kstar / (self.kstar - 1)
            )

            # get diagonal elements of the covariance matrix
            self.grads_var = np.zeros((self.N, self.dims))
            for i in range(self.N):
                self.grads_var[i, :] = np.diag(self.grads_covmat[i, :, :])

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing gradients".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_deltaFs(self, pearson_method="jaccard", comp_p_mat=False):
        """Compute deviations deltaFij to standard kNN log-densities at point j as seen from point i using
            a linear expansion with as slope the semisum of the average gradient of the log-density over
            the neighbourhood of points i and j.

            If not defined, compute the Pearson coefficients p (see docs for pearson_array and pearson_mat) by running
            compute_pearson.
            Then use these p in the estimate of the variances on the deltaFij as 1/4*(E_i^2+E_j^2+2*E_i*E_j*chi), where
            E_i is the error on the estimate of grad_i*DeltaX_ij (see [Carli2024]).
            The log-density differences are stored Fij_array, their variances in Fij_array_var.

        Args:
            pearson_method: see docs for compute_pearson function
            comp_p_mat: see docs for compute_pearson function

        """

        if self.grads_covmat is None:
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
        g_var1 = self.grads_covmat[self.nind_list[:, 0]]
        g_var2 = self.grads_covmat[self.nind_list[:, 1]]

        # check or compute common_neighs
        if self.pearson_mat is None:
            self.compute_pearson(method=pearson_method, comp_p_mat=comp_p_mat)

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
        self.Fij_var_array = 0.25 * (
            vari + varj + 2 * self.pearson_array * np.sqrt(vari * varj)
        )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing gradient corrections".format(sec2 - sec))

        self.Fij_array = Fij_array
        self.Fij_var_array = self.Fij_var_array

    # ----------------------------------------------------------------------------------------------

    def compute_deltaFs_inv_cross_covariance(self, pearson_method="jaccard"):
        """Compute the appoximate inverse cross-covariance of the deltaFs cov[deltaFij,deltaFlm] using the LSDI
        approximation (see compute_density_BMTI_reg docs)

        Args:
            pearson_method: see docs for compute_pearson function

        """

        # check for deltaFs
        if self.pearson_mat is None:
            self.compute_pearson(method=pearson_method, comp_p_mat=True)

        # check or compute deltaFs_grads_semisum
        if self.Fij_var_array is None:
            self.compute_deltaFs()

        if self.verb:
            print("Estimation of the deltaFs cross-covariance started")
        sec = time.time()
        # compute a diagonal approximation of the inverse of the cross-covariance matrix
        self.inv_deltaFs_cov = cgr.return_deltaFs_inv_cross_covariance(
            self.grads_var,
            self.neigh_vector_diffs,
            self.nind_list,
            self.pearson_mat,
            self.Fij_var_array,
        )

        sec2 = time.time()
        if self.verb:
            print(
                "{0:0.2f} seconds computing the deltaFs cross-covariance".format(
                    sec2 - sec
                )
            )

    # ----------------------------------------------------------------------------------------------

    def compute_density_BMTI(
        self,
        delta_F_inv_cov="uncorr",
        comp_log_den_err=False,
        mem_efficient=False,
        alpha=1,
        log_den=None,
        log_den_err=None,
    ):
        """Compute the log-density for each point using BMTI.

        If alpha<1, the algorithm also includes a regularisatin. The regulariser log-density and its errors can be
        passed as arguments: log_den and log_den_err. If any of these two is not specified, use kstarNN estimator
        as a regulariser.

        Args:
            delta_F_inv_cov (str): specify the method used to invert the cross-covariance matrix C of the log-density
                deviations cov[deltaF_ij,deltaF_kl]. Currently implemented methods:
                    "uncorr" (default): all the deltaFs are assumed uncorrelated, i.e. C is assumed to be diagonal with
                        diagonal = Fij_var_array
                    "identity": C is assumed as the identity matrix, so that all terms in the BMTI likelihood are taken
                        unweighted (variance of deltaF_ij = 1 for all (i,j) couples)
                    "LSDI":  (Least Squares with respect to a Diagonal Inverse). Invert the cross-covariance C by
                        finding the approximate diagonal inverse which multiplied by C gives the least-squares closest
                        matrix to the identity in the Frobenius norm
            comp_log_den_err (bool): if True, compute the error on the BMTI estimates. Can be highly time consuming
            mem_efficient (bool): if True, use a sparse matrice to solve BMTI linear system (slower). If False, use a
                dense NxN matrix; this is faster, but can require a great amount of memory if the system is large.
            alpha (float): can take values from 0.0 to 1.0. Indicates the portion of BMTI in the sum of the likelihoods
                alpha*L_BMTI + (1-alpha)*L_kstarNN. Setting alpha=1.0 corresponds to not reguarising BMTI.
            log_den (np.ndarray(float)): size N. The array of the log-densities of the regulariser.
            log_den_err (np.ndarray(float)): size N. The array of the log-density errors of the regulariser.

        """

        # compute changes in free energy
        if self.Fij_array is None:
            self.compute_deltaFs()

        # note: this should be called after the computation of the deltaFs
        # since otherwhise self.log_den and self.log_den_err are redefined to None via set kstar
        if log_den is not None and log_den_err is not None:
            self.log_den = log_den
            self.log_den_err = log_den_err
        else:
            log_den, log_den_err, _ = return_not_normalised_density_kstarNN(
                self.distances,
                self.intrinsic_dim,
                self.kstar,
                interpolation=False,
                bias=False,
            )
            # Normalise density
            log_den -= np.log(self.N)
            self.log_den = log_den
            self.log_den_err = log_den_err

        # add a warnings.warning if self.N > 10000 and mem_efficient is False
        if self.N > 15000 and mem_efficient is False:
            warnings.warn(
                "The number of points is large and the memory efficient option is not selected. \
                If you run into memory issues, consider using the slower memory efficient option."
            )

        if self.verb:
            print("BMTI density estimation started")
            sec = time.time()

        # define the likelihood covarince matrix
        A, deltaFcum = self._get_BMTI_reg_linear_system(delta_F_inv_cov, alpha)

        sec2 = time.time()

        if self.verb:
            print("{0:0.2f} seconds to fill sparse matrix".format(sec2 - sec))

        # solve linear system
        log_den = self._solve_BMTI_reg_linar_system(A, deltaFcum, mem_efficient)
        self.log_den = log_den

        if self.verb:
            print("{0:0.2f} seconds to solve linear system".format(time.time() - sec2))
        sec2 = time.time()

        # compute error
        if comp_log_den_err is True:
            A = A.todense()
            B = slin.pinvh(A)
            self.log_den_err = np.sqrt(np.diag(B))

            if self.verb:
                print("{0:0.2f} seconds inverting A matrix".format(time.time() - sec2))

            sec2 = time.time()

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds for BMTI density estimation".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def _get_BMTI_reg_linear_system(self, delta_F_inv_cov, alpha):
        if delta_F_inv_cov == "uncorr":
            # define redundancy factor for each A matrix entry as the geometric mean of the 2 corresponding kstar
            k1 = self.kstar[self.nind_list[:, 0]]
            k2 = self.kstar[self.nind_list[:, 1]]
            redundancy = np.sqrt(k1 * k2)

            tmpvec = (
                np.ones(self.nspar, dtype=np.float_) / self.Fij_var_array / redundancy
            )
        elif delta_F_inv_cov == "LSDI":
            self.compute_deltaFs_inv_cross_covariance()
            tmpvec = self.inv_deltaFs_cov

        elif delta_F_inv_cov == "identity":
            tmpvec = np.ones(self.nspar, dtype=np.float_)

        else:
            raise ValueError(
                "The delta_F_inv_cov parameter is not valid, choose 'uncorr', 'LSDI' or 'none'"
            )

        # compute adjacency matrix
        A = sparse.csr_matrix(
            (-tmpvec, (self.nind_list[:, 0], self.nind_list[:, 1])),
            shape=(self.N, self.N),
            dtype=np.float_,
        )

        # compute coefficients vector
        supp_deltaF = sparse.csr_matrix(
            (self.Fij_array * tmpvec, (self.nind_list[:, 0], self.nind_list[:, 1])),
            shape=(self.N, self.N),
            dtype=np.float_,
        )

        # make A symmetric
        A = alpha * sparse.lil_matrix(A + A.transpose())

        # insert kstarNN with factor 1-alpha in the Gaussian approximation
        # ALREADY MULTIPLIED A BY ALPHA
        diag = (
            np.array(-A.sum(axis=1)).reshape((self.N,))
            + (1.0 - alpha) / self.log_den_err**2
        )

        A.setdiag(diag)

        deltaFcum = (
            alpha
            * (
                np.array(supp_deltaF.sum(axis=0)).reshape((self.N,))
                - np.array(supp_deltaF.sum(axis=1)).reshape((self.N,))
            )
            + (1.0 - alpha) * self.log_den / self.log_den_err**2
        )

        return A, deltaFcum

    def _solve_BMTI_reg_linar_system(self, A, deltaFcum, mem_efficient):
        if mem_efficient is False:
            log_den = np.linalg.solve(A.todense(), deltaFcum)
        else:
            log_den = sparse.linalg.spsolve(A.tocsr(), deltaFcum)

        return log_den
