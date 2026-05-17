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
The *kstar* module contains the *KStar* class.

The computation of the optimal neighbourhood size (k*) is implemented in this class as the compute_kstar method.
"""

import time
import warnings

import numpy as np

from dadapy._cython import cython_density as cd
from dadapy._utils import utils as ut
from dadapy._utils.utils import cores
from dadapy.id_estimation import IdEstimation


class KStar(IdEstimation):
    """Computes for each point an optimal choice - kstar - of the neighbourhood size.

    Inherits from class IdEstimation.
    Can assign to the data a user-defined neighbourhood size.

    Attributes:
        kstar (np.array(float)): array containing the chosen number k* in the neighbourhood of each of the N points
        dc (np.array(float), optional): array containing the distance of the k*th neighbor from each of the N points
    """

    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        n_jobs=cores,
        rng_seed=42,
    ):
        """Initialise the KStar class."""
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
            verbose=verbose,
            n_jobs=n_jobs,
            rng_seed=rng_seed,
        )

        self.kstar = None
        self.dc = None

    # ----------------------------------------------------------------------------------------------

    def reset_kstar(self):
        """Set kstar and dc to None."""
        self.kstar = None
        self.dc = None

    # ----------------------------------------------------------------------------------------------

    def set_kstar(self, k=0):
        """Set all elements of kstar to a specified value k.

        Invokes reset_kstar.

        Args:
            k: number of neighbours used to compute the density it can be an iteger or an array of integers
        """
        self.reset_kstar()

        # raise warning if self.intrinsic_dim is None using the warning module
        if self.intrinsic_dim is None:
            warnings.warn(
                "Setting the k value but, be careful: the intrinsic dimension is not defined!",
                stacklevel=2,
            )

        if isinstance(k, np.ndarray):
            self.kstar = k
        else:
            self.kstar = np.full(self.N, k, dtype=int)

    # ----------------------------------------------------------------------------------------------

    def compute_kstar(self, alpha=1e-6, bonferroni_deloc=False, bonferroni_loc=False):
        """Compute an optimal choice of the neighbourhood size k for each point.

        Args:
            alpha (float): Likelihood ratio parameter used to compute optimal k, i.e. quantile
                for the unfirm density likelihood-ratio test.
            bonferroni_deloc (bool): apply bonferroni correction for multiple testing across
                the dataset
            bonferroni_loc (bool): apply bonferroni correction for multiple testing correcting
                the threshold at each iteration

        """
        if self.intrinsic_dim is None:
            warnings.warn(
                "Careful! The intrinsic dimension is not defined. "
                "Computing it unsupervisedly with 'compute_id_2NN()' method",
                stacklevel=2,
            )
            _ = self.compute_id_2NN()

        if self.verb:
            print(f"kstar estimation started, alpha = {alpha}")

        sec = time.time()

        kstar = cd._compute_kstar(
            self.intrinsic_dim,
            self.N,
            self.maxk,
            alpha,
            self.dist_indices.astype("int64"),
            self.distances.astype("float64"),
            bonferroni_deloc,
            bonferroni_loc,
        )

        self.set_kstar(kstar)

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing kstar".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def return_ids_kstar_gride(
        self,
        initial_id=None,
        n_iter=5,
        alpha=1e-6,
        d0=0.001,
        d1=1000,
        eps=1e-7,
        bonferroni_deloc=False,
        bonferroni_loc=False,
    ):
        """Return the id estimates of the Gride algorithm coupled with the kstar estimation of the scale.

        Args:
            initial_id (float): initial estimate of the id default uses 2NN
            n_iter (int): number of iteration
            alpha (float): threshold value for the kstar test
            d0 (float): minimum id value
            d1 (float): maximum id value
            eps (float): threshold for the convergence of the Gride algorithm
            bonferroni_deloc (bool): apply bonferroni correction for multiple testing across the dataset
            bonferroni_loc (bool): apply bonferroni correction for multiple testing correcting the threshold
             at each iteration

        Returns:
            ids, ids_err, kstars, log_likelihoods
        """
        # start with an initial estimate of the ID
        if initial_id is None:
            self.compute_id_2NN()
        else:
            self.set_id(initial_id)
            if self.distances is None:
                self.compute_distances()
        # compute kstar
        self.compute_kstar(alpha, bonferroni_deloc, bonferroni_loc)

        ids = [self.intrinsic_dim]
        ids_err = [self.intrinsic_dim_err]
        kstars = [self.kstar]
        log_likelihoods = [0]

        for i in range(n_iter):
            print("iteration ", i)
            print("id ", self.intrinsic_dim)

            # compute n2 and n1 via kstar. If not even, make it even by adding one
            n2s = self.kstar
            not_even = n2s % 2 != 0
            n2s[not_even] = n2s[not_even] + 1
            assert sum(n2s % 2 != 0) == 0
            n1s = (n2s / 2).astype(int)

            # compute the mus
            mus = np.array(
                [
                    self.distances[i, n2] / self.distances[i, n1]
                    for i, (n1, n2) in enumerate(zip(n1s, n2s))
                ]
            )
            # compute the id using Gride
            gride_id, id_err = self._compute_id_gride_single_scale(
                d0, d1, mus, n1s, n2s, eps
            )
            self.set_id(gride_id)
            log_lik = -ut._neg_loglik(self.dtype, gride_id, mus, n1s, n2s)
            self.compute_kstar(alpha, bonferroni_deloc, bonferroni_loc)

            ids.append(gride_id)
            ids_err.append(id_err)
            kstars.append(self.kstar)
            log_likelihoods.append(log_lik)

        ids = np.array(ids)
        ids_err = np.array(ids_err)
        kstars = np.array(kstars)
        log_likelihoods = np.array(log_likelihoods)

        id_scale = 0.0
        for i, (n1, n2) in enumerate(zip(n1s, n2s)):
            id_scale += self.distances[i, n1]
            id_scale += self.distances[i, n2]
        id_scale /= 2 * self.N

        self.intrinsic_dim = gride_id
        self.intrinsic_dim_err = id_err
        self.intrinsic_dim_scale = id_scale

        return ids, ids_err, kstars, log_likelihoods

    # ----------------------------------------------------------------------------------------------

    def return_ids_kstar_binomial(
        self,
        initial_id=None,
        n_iter=5,
        alpha=1e-6,
        bonferroni_deloc=False,
        bonferroni_loc=False,
        r=None,
        plot_mv=False,
        k_bootstrap=1,
    ):
        """Return the id estimates of the binomial algorithm coupled with the kstar estimation of the scale.

        Args:
            initial_id (float): initial estimate of the id default uses 2NN
            n_iter (int): number of iteration
            alpha (float): threshold value for the kstar test
            bonferroni_deloc (bool): apply bonferroni correction for multiple testing across the dataset
            bonferroni_loc (bool): apply bonferroni correction for multiple testing correcting the threshold
             at each iteration
            r (float, default=None): parameter of binomial estimator, 0 < r < 1.
             If None, the optimal, adaptive one is used
            plot_mv (bool, default=False): if True, plots the observed and the theoretical distributions
             of the number of points in the shells
            k_bootstrap (int, default=1): number of bootstrap resampling to estimate the pvalue of the ID estimation

        Returns:
            ids (np.ndarray(float)): intrinsic dimension across iterations
            ids_err (np.ndarray(float)): intrinsic dimension error across iterations
            kstars (np.ndarray(int): arrays of kstars across iterations
            p-values (np.ndarray(float)): p-values from model validation across iterations
        """
        # start with an initial estimate of the ID and the associated k*
        if initial_id is None:
            self.compute_id_2NN(algorithm="base")
        else:
            self.set_id(initial_id)
            if self.distances is None:
                self.compute_distances()
        self.compute_kstar(alpha, bonferroni_deloc, bonferroni_loc)

        ids = [self.intrinsic_dim]
        ids_err = [self.intrinsic_dim_err]
        kstars = [self.kstar]
        pvalues = [0]

        for i in range(n_iter):
            print("iteration ", i)
            print("id ", self.intrinsic_dim)

            # set new ratio
            r_eff = min(0.975, 0.2032 ** (1.0 / self.intrinsic_dim)) if r is None else r
            # compute id using the k*
            ide, id_err, _, pv = self.compute_id_binomial_k(
                self.kstar, r_eff, bayes=False, plot_mv=plot_mv, k_bootstrap=k_bootstrap
            )
            # compute likelihood
            """
            n = self._fix_k(self.kstar, r_eff)
            log_lik = ut.binomial_loglik(ide, self.kstar - 1, n - 1, r_eff)
            """

            # update the k*
            self.compute_kstar(alpha, bonferroni_deloc, bonferroni_loc)
            # store the obtained values
            ids.append(ide)
            ids_err.append(id_err)
            kstars.append(self.kstar)
            pvalues.append(pv)

        ids = np.array(ids)
        ids_err = np.array(ids_err)
        kstars = np.array(kstars)
        pvalues = np.array(pvalues)

        return ids, ids_err, kstars, pvalues
