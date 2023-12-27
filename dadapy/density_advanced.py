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

Different algorithms to estimate the logdensity, the logdensity gradientest and the logdensity differences are implemented as methods of this class.
In particular, differently from the methods implemented in the DensityEstimation, the methods in the DensityEstimation class are based on the sparse neighbourhood graph structure which is implemented in the NeighGraph class.
"""

import multiprocessing
import time

import numpy as np
from scipy import sparse
from scipy import linalg as slin

from dadapy._cython import cython_density as cd
from dadapy._cython import cython_grads as cgr

from dadapy.neighbourhood_graph import NeighGraph

cores = multiprocessing.cpu_count()


class DensityEstimation(NeighGraph):
    """Computes the log-density and (where implemented) its error at each point and other properties.

    Inherits from class NeighGraph.
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    Can return an estimate of the gradient of the log-density at each point and an estimate of the error on each component using an improved version of the mean-shift gradient algorithm [Fukunaga1975][Carli2023]
    Can return an estimate of log-density differences and their error each point based on the gradient estimates.
    Can compute the log-density and its error at each using BMTI.
    

    Attributes:
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

        self.grads = None
        self.grads_var = None
        self.check_grads_covmat = False
        self.Fij_array = None
        self.Fij_var_array = None
        self.Fij_var_array = None
        inv_deltaFs_cov = None

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

            AAAAAAAAAAAAAAAA possibile spostarlo in utils al momento. Peraltro qui bisogna trovare un modo per farlo funzionare

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


    def compute_density_BMTI(self, method=uncorr, use_variance=True, redundancy_factor=None, comp_err=False, mem_efficient=True):

        # method    = uncorr assumes the cross-covariance matrix is diagonal with diagonal = Fij_var_array;
        #           = LSDI (Least Squares with respect to a Diagonal Inverse) inverts the cross-covariance C
        #             by finding the approximate diagonal inverse which multiplied by C gives the least-squared
        #             closest matrix to the identity in the Frobenius norm
        # use_variance  = True uses the elements of the inverse cross-covariance to define the A matrix;
        #               = False assumes the cross-covraiance matix is equal to the nspar x nspar identity
        # redundancy_factor (used only if method=uncorr)
        # comp_err
        # mem_efficient = True uses sparse matrices;
        #               = False uses dense NxN matrices


        # compute changes in free energy
        if self.Fij_array is None:
            self.compute_deltaFs_grads_semisum()

        if self.verb:
            print("BMTI density estimation started")
            sec = time.time()

        # define redundancy factor (equals 1 in all cases in which method != uncorr)
        if redundancy_factor is None:
            redundancy = np.ones(self.nspar,dtype=np.float_)

        elif redundancy_factor=='geometric_mean':
            assert method is 'uncorr', "redundancy_factor can be defined only for 'uncorr' method"
            # define redundancy factor for each A matrix entry as the geometric mean of the 2 corresponding k*
            k1 = self.kstar[self.nind_list[:, 0]]
            k2 = self.kstar[self.nind_list[:, 1]]
            redundancy = np.sqrt(k1 * k2)

        elif np.size(redundancy_factor) == self.nspar:
            assert method is uncorr, "redundancy_factor can be defined only for 'uncorr' method"
            # redundancy vector
            redundancy = redundancy_factor

        else
            print("Invalid 'redundancy_factor' value")
            break    

        # define the likelihood covarince matrix
        if use_variance:
            if method is 'uncorr':
                tmpvec = np.ones(self.nspar, dtype=np.float_)/ self.Fij_var_array / redundancy
            elif method is 'LSDI':
                self.compute_deltaFs_inv_cross_covariance()
                tmpvec = self.inv_deltaFs_cov    
        else:
            tmpvec = np.ones(self.nspar, dtype=np.float_)/ redundancy

        sec2 = time.time()


        # compute adjacency matrix and coefficients vector
        A = sparse.csr_matrix((-tmpvec, (self.nind_list[:, 0], self.nind_list[:, 1])), shape=(self.N,self.N), dtype=np.float_)

        supp_deltaF = sparse.csr_matrix(( self.Fij_array * tmpvec, (self.nind_list[:, 0], self.nind_list[:, 1])), shape=(self.N,self.N), dtype=np.float_)

        A = sparse.lil_matrix(A + A.transpose())
        diag = np.array(-A.sum(axis=1)).reshape((self.N,))
        A.setdiag(diag)

        deltaFcum = np.array(supp_deltaF.sum(axis=0)).reshape((self.N,)) - np.array(
            supp_deltaF.sum(axis=1)
        ).reshape((self.N,))

        if self.verb:
            print("{0:0.2f} seconds to fill sparse matrix".format(time.time() - sec2))


        # solve linear system
        sec2 = time.time()
        if mem_efficient==False:
            log_den = np.linalg.solve(A.todense(), deltaFcum)    

        else:
            log_den = sparse.linalg.spsolve(A.tocsr(), deltaFcum)

        self.log_den = log_den

        if self.verb:
            print("{0:0.2f} seconds to solve linear system".format(time.time() - sec2))
        
        sec2 = time.time()


        # compute error
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

    def compute_density_kstarNN_gCorr(
        self,
        use_variance=True,
        gauss_approx=True, #see Jan 2022 version compute_density_PAk_gCorr
        alpha=1.0,
        log_den_kstarNN=None,
        log_den_err_kstarNN=None,
        comp_err=False,
        redundancy_factor=None,
        mem_efficient=True
    ):

        if log_den_kstarNN is not None and log_den_err_kstarNN is not None:
            self.log_den = log_den_kstarNN
            self.log_den_err = log_den_err_kstarNN

        else:
            self.compute_density_kstarNN()

        # compute changes in free energy
        if self.Fij_array is None:
            self.compute_deltaFs_grads_semisum()

        if self.verb:
            print("kastarNN+gCorr density estimation started")
            sec = time.time()


        if redundancy_factor is None:
            redundancy = np.ones(self.nspar,dtype=np.float_)

        elif redundancy_factor=='geometric_mean':
            # define redundancy factor for each A matrix entry as the geometric mean of the 2 corresponding k*
            k1 = self.kstar[self.nind_list[:, 0]]
            k2 = self.kstar[self.nind_list[:, 1]]
            redundancy = np.sqrt(k1 * k2)

        elif np.size(redundancy_factor) == self.nspar:
            # redundancy vector
            redundancy = redundancy_factor    
        
        # compute non-zero A-matrix elements
        if use_variance:
            tmpvec = np.ones(self.nspar, dtype=np.float_)/ self.Fij_var_array / redundancy
        else:
            tmpvec = np.ones(self.nspar, dtype=np.float_)/ redundancy

        # initialise sparse adjacency matrix
        A = sparse.csr_matrix((-tmpvec, (self.nind_list[:, 0], self.nind_list[:, 1])), shape=(self.N,self.N), dtype=np.float_)
        
        # initialise coefficients vector
        supp_deltaF = sparse.csr_matrix(( self.Fij_array * tmpvec, (self.nind_list[:, 0], self.nind_list[:, 1])), shape=(self.N,self.N), dtype=np.float_)


        A = alpha * sparse.lil_matrix(A + A.transpose())

        # insert kstarNN with factor 1-alpha in the Gaussian approximation
        # ALREADY MULTIPLIED A BY ALPHA
        diag = (
            np.array(-A.sum(axis=1)).reshape((self.N,))
            + (1.0 - alpha) / self.log_den_err ** 2
        )
            
        A.setdiag(diag)

        deltaFcum = (
                alpha
                * (
                    np.array(supp_deltaF.sum(axis=0)).reshape((self.N,))
                    - np.array(supp_deltaF.sum(axis=1)).reshape((self.N,))
                )
                + (1.0 - alpha) * self.log_den / self.log_den_err ** 2
            )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds to fill sparse matrix".format(sec2 - sec))

        if mem_efficient==False:
            log_den = np.linalg.solve(A.todense(), deltaFcum)    

        else:
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
            print("{0:0.2f} seconds for kastarNN+gCorr density estimation".format(sec2 - sec))

    # ---------------------------------------------------------------------------------------------- 


