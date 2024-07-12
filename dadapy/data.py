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
The *data* module contains the *Data* class.

Such a class inherits from all other classes defined in the package and as such it provides a convenient container of
all the algorithms implemented in Dadapy.
"""

import multiprocessing
import os

import numpy as np

from dadapy._utils import utils as ut
from dadapy.clustering import Clustering
from dadapy.density_advanced import DensityAdvanced
from dadapy.feature_weighting import FeatureWeighting
from dadapy.metric_comparisons import MetricComparisons

rng = np.random.default_rng()

cores = multiprocessing.cpu_count()
np.set_printoptions(precision=2)
os.getcwd()


class Data(Clustering, DensityAdvanced, MetricComparisons, FeatureWeighting):
    """Data class."""

    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        verbose=False,
        n_jobs=cores,
        working_memory=1024,
    ):
        """Initialise a Data object, container of all DADApy methods.

        It is initialised with a set of coordinates or a set of
        distances, and all methods can be called on the generated class instance.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            distances (np.ndarray(float)): A matrix of dimension N x mask containing distances between points
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            verbose (bool): whether you want the code to speak or shut up
            n_jobs (int): number of cores to be used
            working_memory (int): working memory (TODO: currently unused)
        """
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    def return_ids_kstar_gride(
        self, initial_id=None, n_iter=5, Dthr=23.92812698, d0=0.001, d1=1000, eps=1e-7
    ):
        """Return the id estimates of the Gride algorithm coupled with the kstar estimation of the scale.

        Args:
            initial_id: initial estimate of the id default uses 2NN
            n_iter: number of iteration
            Dthr: threshold value for the kstar test
            d0: minimum id value
            d1: maximum id value
            eps: threshold for the convergence of the Gride algorithm

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
        self.compute_kstar(Dthr)

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
            id, id_err = self._compute_id_gride_single_scale(d0, d1, mus, n1s, n2s, eps)
            self.set_id(id)
            log_lik = -ut._neg_loglik(self.dtype, id, mus, n1s, n2s)
            self.compute_kstar(Dthr)

            ids.append(id)
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

        self.intrinsic_dim = id
        self.intrinsic_dim_err = id_err
        self.intrinsic_dim_scale = id_scale

        return ids, ids_err, kstars, log_likelihoods

    def return_ids_kstar_binomial(
        self,
        initial_id=None,
        n_iter=5,
        Dthr=23.92812698,
        r=None,
        plot_mv=False,
        k_bootstrap=1,
    ):
        """Return the id estimates of the binomial algorithm coupled with the kstar estimation of the scale.

        Args:
            initial_id (float): initial estimate of the id default uses 2NN
            n_iter (int): number of iteration
            Dthr (float): threshold value for the kstar test
            r (float, default=None): parameter of binomial estimator, 0 < r < 1. If None, the optimal, adaptive one is
             used
            plot_mv (bool, default=False): if True, plots the observed and the theoretical distributions

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
        self.compute_kstar(Dthr)

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
            ide, id_err, scale, pv = self.compute_id_binomial_k(
                self.kstar, r_eff, bayes=False, plot_mv=plot_mv, k_bootstrap=k_bootstrap
            )
            # compute likelihood
            """
            n = self._fix_k(self.kstar, r_eff)
            log_lik = ut.binomial_loglik(ide, self.kstar - 1, n - 1, r_eff)
            """

            # update the k*
            self.compute_kstar(Dthr)
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
