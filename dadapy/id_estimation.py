# Copyright 2021-2022 The DADApy Authors. All Rights Reserved.
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
The *id_estimation* module contains the *IdEstimation* class.

The different algorithms of intrinsic dimension estimation are implemented as methods of this class.
"""
import copy
import math
import multiprocessing
import warnings
from functools import partial

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import pairwise_distances_chunked

from dadapy._utils import utils as ut
from dadapy._utils.utils import compute_nn_distances
from dadapy.base import Base

cores = multiprocessing.cpu_count()
rng = np.random.default_rng()


class IdEstimation(Base):
    """IdEstimation class."""

    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        njobs=cores,
    ):
        """Estimate the intrinsic dimension of a dataset choosing among various routines.

        Inherits from class Base.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            distances (np.ndarray(float)): A matrix of dimension N x mask containing distances between points
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            period (np.array(float), optional): array containing periodicity for each coordinate. Default is None
            verbose (bool): whether you want the code to speak or shut up
            njobs (int): number of cores to be used
        """
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
            verbose=verbose,
            njobs=njobs,
        )

        self.intrinsic_dim = None
        self.intrinsic_dim_err = None
        self.intrinsic_dim_scale = None
        self.intrinsic_dim_mus = None
        self.intrinsic_dim_mus_gride = None

    # ----------------------------------------------------------------------------------------------

    def _compute_id_2NN(self, mus, fraction, algorithm="base"):
        """Compute the id using the 2NN algorithm.

        Helper of return return_id_2NN.

        Args:
            mus (np.ndarray(float)): ratio of the distances of first- and second-nearest neighbours
            fraction (float): fraction of mus to take into account, discard the highest values
            algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood

        Returns:
            intrinsic_dim (float): the estimation of the intrinsic dimension
        """
        N = mus.shape[0]
        N_eff = int(N * fraction)
        log_mus = np.log(mus)
        log_mus_reduced = np.sort(log_mus)[:N_eff]

        if algorithm == "ml":
            intrinsic_dim = (N - 1) / np.sum(log_mus)

        elif algorithm == "base":
            y = -np.log(1 - np.arange(1, N_eff + 1) / N)

            def func(x, m):
                return m * x

            intrinsic_dim, _ = curve_fit(func, log_mus_reduced, y)

        else:
            raise ValueError("Please select a valid algorithm type")

        return intrinsic_dim

    # ----------------------------------------------------------------------------------------------
    def compute_id_2NN(
        self,
        algorithm="base",
        fraction=0.9,
        decimation=1,
        set_attr=True,
    ):
        """Compute intrinsic dimension using the 2NN algorithm.

        Args:
            algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood
            fraction (float): fraction of mus that will be considered for the estimate (discard highest mus)
            decimation (float): fraction of randomly sampled points used to compute the id
            set_attr (bool): whether to change the class attributes as a result of the computation

        Returns:
            id (float): the estimated intrinsic dimension
            id_err (float): the standard error on the id estimation
            rs (float): the average nearest neighbor distance (rs)

        Quick Start:
        ===========

        .. code-block:: python

                from dadapy import Data
                from sklearn.datasets import make_swiss_roll

                n_samples = 5000
                X, _ = make_swiss_roll(n_samples, noise=0.0)

                ie = Data(coordinates=X)

                results = ie.compute_id_2NN()
                results:
                (1.96, 0.0, 0.38)       #(id, error, average distance to the first two neighbors)

                results = ie.compute_id_2NN(fraction = 1)
                results:
                (1.98, 0.0, 0.38)       #(id, error, average distance to the first two neighbors)

                results = ie.compute_id_2NN(decimation = 0.25)
                results:
                (1.99, 0.036, 0.76)     #(id, error, average distance to the first two neighbors)
                                        #1/4 of the points are kept.
                                        #'id' is the mean over 4 bootstrap samples;
                                        #'error' is standard error of the sample mean.

        References:
            E. Facco, M. d’Errico, A. Rodriguez, A. Laio, Estimating the intrinsic dimension of datasets by a minimal
            neighborhood information, Scientific reports 7 (1) (2017) 1–8
        """
        assert (
            0.0 < decimation and decimation <= 1.0
        ), "'decimation' must be between 0 and 1"
        assert 0.0 < fraction and fraction <= 1.0, "'fraction' must be between 0 and 1"
        if fraction == 1.0 and algorithm == "base":
            algorithm = "ml"
            print("fraction = 1: algorithm set to ml")

        nrep = int(np.rint(1.0 / decimation))
        ids = np.zeros(nrep)
        rs = np.zeros(nrep)

        N_subset = int(np.rint(self.N * decimation))
        mus = np.zeros((N_subset, nrep))

        for j in range(nrep):

            if decimation == 1 and self.distances is not None:
                # with decimation == 1 use saved distances if present
                distances, dist_indices = self.distances, self.dist_indices

            elif decimation == 1 and self.distances is None and set_attr is True:
                # with decimation ==1 and set_attr==True compute distances and save them
                self.compute_distances()
                distances, dist_indices = self.distances, self.dist_indices

            else:
                # if set_attr==False or for decimation < 1 random sample points don't save distances
                idx = np.random.choice(self.N, size=N_subset, replace=False)
                X_decimated = self.X[idx]

                distances, dist_indices = compute_nn_distances(
                    X_decimated,
                    maxk=3,  # only compute first 2 nn
                    metric=self.metric,
                    period=self.period,
                )

            mus[:, j] = distances[:, 2] / distances[:, 1]
            ids[j] = self._compute_id_2NN(mus[:, j], fraction, algorithm)
            rs[j] = np.mean(distances[:, np.array([1, 2])])

        intrinsic_dim = np.mean(ids)
        intrinsic_dim_err = np.std(ids) / len(ids) ** 0.5
        intrinsic_dim_scale = np.mean(rs)

        if self.verb:
            print(f"ID estimation finished: selecting ID of {intrinsic_dim}")

        if set_attr:
            self.intrinsic_dim = intrinsic_dim
            self.intrinsic_dim_err = intrinsic_dim_err
            self.intrinsic_dim_scale = intrinsic_dim_scale
            self.intrinsic_dim_mus = mus

        return intrinsic_dim, intrinsic_dim_err, intrinsic_dim_scale

    # ----------------------------------------------------------------------------------------------

    def return_id_scaling_2NN(
        self,
        N_min=10,
        algorithm="base",
        fraction=0.9,
    ):
        """Compute the id at different scales using the 2NN algorithm.

        Args:
            N_min (int): minimum number of points considered when decimating the dataset,
                        N_min effectively sets the largest 'scale';
            algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood;
            fraction (float): fraction of mus that will be considered for the estimate (discard highest mus).

        Returns:
            ids_scaling (np.ndarray(float)): array of intrinsic dimensions;
            ids_scaling_err (np.ndarray(float)): array of error estimates;
            rs_scaling (np.ndarray(float)): array of average distances of the neighbors involved in the estimates.

        Quick Start:
        ===========

        .. code-block:: python

                from dadapy import Data
                from sklearn.datasets import make_swiss_roll

                #two dimensional curved manifold embedded in 3d with noise

                n_samples = 5000
                X, _ = make_swiss_roll(n_samples, noise=0.3)

                ie = Data(coordinates=X)
                ids_scaling, ids_scaling_err, rs_scaling = ie.return_id_scaling_2NN(N_min = 20)

                ids_scaling:
                array([2.88 2.77 2.65 2.42 2.22 2.2  2.1  2.23])

                ids_scaling_err:
                array([0.   0.02 0.05 0.04 0.04 0.03 0.04 0.04])

                rs_scaling:
                array([0.52 0.66 0.88 1.18 1.65 2.3  3.23 4.54])
        """
        max_ndec = int(math.log(self.N, 2)) - 1
        Nsubsets = np.round(self.N / np.array([2**i for i in range(max_ndec)]))
        Nsubsets = Nsubsets.astype(int)

        if N_min is not None:
            Nsubsets = Nsubsets[Nsubsets > N_min]

        ids_scaling = np.zeros(Nsubsets.shape[0])
        ids_scaling_err = np.zeros(Nsubsets.shape[0])
        rs_scaling = np.zeros((Nsubsets.shape[0]))

        for i, N_subset in enumerate(Nsubsets):

            ids_scaling[i], ids_scaling_err[i], rs_scaling[i] = self.compute_id_2NN(
                algorithm=algorithm,
                fraction=fraction,
                decimation=N_subset / self.N,
                set_attr=False,
            )

        return ids_scaling, ids_scaling_err, rs_scaling

    # ----------------------------------------------------------------------------------------------
    def return_id_scaling_gride(
        self, range_max=64, d0=0.001, d1=1000, eps=1e-7, save_mus=False
    ):
        """Compute the id at different scales using the Gride algorithm.

        Args:
            range_max (int): maximum nearest neighbor rank considered for the id computations;
                            the number of id estimates are log2(range_max) as the nearest neighbor
                            order ('scale') is doubled at each estimate;
            d0 (float): minimum intrinsic dimension considered in the search;
            d1 (float): maximum intrinsic dimension considered in the search;
            eps (float): precision of the approximate id calculation.

        Returns:
            ids_scaling (np.ndarray(float)): array of intrinsic dimensions of length log2(range_max);
            ids_scaling_err (np.ndarray(float)): array of error estimates;
            rs_scaling (np.ndarray(float)): array of average distances of the neighbors involved in the estimates.

        Quick Start:
        ===========

        .. code-block:: python

                from dadapy import Data
                from sklearn.datasets import make_swiss_roll

                #two dimensional curved manifold embedded in 3d with noise

                n_samples = 5000
                X, _ = make_swiss_roll(n_samples, noise=0.3)

                ie = Data(coordinates=X)
                ids_scaling, ids_scaling_err, rs_scaling = ie.return_id_scaling_gride(range_max = 512)

                ids_scaling:
                array([2.81 2.71 2.48 2.27 2.11 1.98 1.95 2.05])

                ids_scaling_err:
                array([0.04 0.03 0.02 0.01 0.01 0.01 0.   0.  ])

                rs_scaling:
                array([0.52 0.69 0.93 1.26 1.75 2.48 3.54 4.99])


        References:
            F. Denti, D. Doimo, A. Laio, A. Mira, Distributional results for model-based intrinsic dimension
            estimators, arXiv preprint arXiv:2104.13832 (2021).
        """
        max_rank = min(self.N, range_max)
        max_step = int(math.log(max_rank, 2))
        nn_ranks = np.array([2**i for i in range(max_step)])

        if self.distances is not None and range_max < self.maxk + 1:
            max_rank = min(max_rank, self.maxk + 1)
            if self.verb:
                print(
                    f"distance already computed up to {max_rank}. max rank set to {max_rank}"
                )

            mus = self.distances[:, nn_ranks[1:]] / self.distances[:, nn_ranks[:-1]]
            rs = self.distances[:, np.array([nn_ranks[:-1], nn_ranks[1:]])]

        elif self.X is not None:

            if self.verb:
                print(
                    f"distance not computed up to {max_rank}. distance computation started"
                )

            distances, dist_indices, mus, rs = self._return_mus_scaling(
                range_scaling=max_rank
            )
            # returns:
            # distances, dist_indices (self.N, self.maxk+1): sorted distances and dist indices up to maxk+1
            # mus (self.N, len(nn_ranks)): ratio between 2*kth and kth neighbor distances of every data point
            # rs (self.N, 2, len(nn_ranks)): kth, 2*kth neighbor of every data, for every nn_ranks
            if self.verb:
                print("distance computation finished")

            # if distances have not been computed save them
            if self.distances is None:
                self.distances = distances
                self.dist_indices = dist_indices
                self.N = distances.shape[0]

        # compute IDs (and their error) via maximum likelihood for all the scales up to max_rank
        if self.verb:
            print("id inference started")
        ids_scaling, ids_scaling_err = self._compute_id_gride(mus, d0, d1, eps)
        if self.verb:
            print("id inference finished")

        "average of the kth and 2*kth neighbor distances taken over all datapoints for each id estimate"
        rs_scaling = np.mean(rs, axis=(0, 1))

        if save_mus:
            self.intrinsic_dim_mus_gride = mus

        return ids_scaling, ids_scaling_err, rs_scaling

    # ----------------------------------------------------------------------------------------------
    def _compute_id_gride(self, mus, d0, d1, eps):
        """Compute the id using the gride algorithm.

        Helper of return return_id_gride.

        Args:
            mus (np.ndarray(float)): ratio of the distances of nth and 2nth nearest neighbours (Ndata x log2(range_max))
            d0 (float): minimum intrinsic dimension considered in the search;
            d1 (float): maximum intrinsic dimension considered in the search;
            eps (float): precision of the approximate id calculation.

        Returns:
            intrinsic_dim (np.ndarray(float): array of id estimates
            intrinsic_dim_err (np.ndarray(float): array of error estimates
        """
        # array of ids (as a function of the average distance to a point)
        ids_scaling = np.zeros(mus.shape[1])
        # array of error estimates (via fisher information)
        ids_scaling_err = np.zeros(mus.shape[1])
        for i in range(mus.shape[1]):
            n1 = 2**i
            id = ut._argmax_loglik(
                self.dtype, d0, d1, mus[:, i], n1, 2 * n1, self.N, eps=eps
            )  # eps=precision id calculation
            ids_scaling[i] = id

            ids_scaling_err[i] = (
                1
                / ut._fisher_info_scaling(
                    id, mus[:, i], n1, 2 * n1, eps=5 * self.eps
                )  # eps=regularization small numbers
            ) ** 0.5

        return ids_scaling, ids_scaling_err

    def _mus_scaling_reduce_func(self, dist, start, range_scaling):
        """Help to compute the "mus" needed to compute the id.

        Applied at the end of pairwise_distance_chunked see:
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/pairwise.py#L1474

        Once a chunk of the distance matrix is computed _mus_scaling_reduce_func
        1) extracts the distances of the  neighbors of order 2**i up to the maximum
        neighbor range given by range_scaling
        2) computes the mus[i] (ratios of the neighbor distance of order 2**(i+1)
        and 2**i (see return id scaling gride)
        3) returns the chunked distances up to maxk, the mus, and rs, the distances
        of the neighbors involved in the estimate

        Args:
            dist: chunk of distance matrix passed internally by pairwise_distance_chunked
            start: dummy variable neede for compatibility with sklearn, not used
            range_scaling (int): maximum neighbor rank

        Returns:
            dist: CHUNK of distance matrix sorted in increasing order of neighbor distances up to maxk
            neighb_ind: indices of the nearest neighbors up to maxk
            mus: ratios of the neighbor distances of order 2**(i+1) and 2**i
            rs: distances of the neighbors involved in the mu estimates
        """
        # argsort may be faster than argpartition when gride is applied on the full dataset (for the moment not used)
        max_step = int(math.log(range_scaling, 2))
        steps = np.array([2**i for i in range(max_step)])

        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, steps[-1], axis=1)
        neigh_ind = neigh_ind[:, : steps[-1] + 1]

        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]

        dist = np.sqrt(dist[sample_range, neigh_ind])
        dist = self._remove_zero_dists(dist)
        mus = dist[:, steps[1:]] / dist[:, steps[:-1]]
        rs = dist[:, np.array([steps[:-1], steps[1:]])]

        dist = copy.deepcopy(dist[:, : self.maxk + 1])
        neigh_ind = copy.deepcopy(neigh_ind[:, : self.maxk + 1])

        return dist, neigh_ind, mus, rs

    def _return_mus_scaling(self, range_scaling):
        """Return the "mus" needed to compute the id.

        Adapted from kneighbors function of sklearn
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/neighbors/_base.py#L596
        It allows to keep a nearest neighbor matrix up to rank 'maxk' (few tens of points)
        instead of 'range_scaling' (few thousands), while computing the ratios between neighbors' distances
        up to neighbors' rank 'range scaling'.
        For big datasets it avoids out of memory errors

        Args:
            range_scaling (int): maximum neighbor rank considered in the computation of the mu ratios

        Returns:
            dist (np.ndarray(float)): the FULL distance matrix sorted in increasing order of distances up to maxk
            neighb_ind np.ndarray(int)): the FULL matrix of the indices of the nearest neighbors up to maxk
            mus np.ndarray(float)): the FULL matrix of the ratios of the neighbor distances of order 2**(i+1) and 2**i
            rs np.ndarray(float)): the FULL matrix of the distances of the neighbors involved in the mu estimates
        """
        reduce_func = partial(
            self._mus_scaling_reduce_func, range_scaling=range_scaling
        )

        kwds = {"squared": True}
        chunked_results = list(
            pairwise_distances_chunked(
                self.X,
                self.X,
                reduce_func=reduce_func,
                metric=self.metric,
                n_jobs=self.njobs,
                working_memory=1024,
                **kwds,
            )
        )

        neigh_dist, neigh_ind, mus, rs = zip(*chunked_results)

        zero_dists = np.sum(
            neigh_dist[0][:, 1] <= 1.1 * np.finfo(neigh_dist[0].dtype).eps
        )
        if zero_dists > 0:
            warnings.warn(
                """there may be data with zero distance from each other;
                this may compromise the correct behavior of some routines"""
            )

        return (
            np.vstack(neigh_dist),
            np.vstack(neigh_ind),
            np.vstack(mus),
            np.vstack(rs),
        )

    # ----------------------------------------------------------------------------------------------

    def compute_id_2NN_wprior(self, alpha=2, beta=5, posterior_mean=True):
        """Compute the intrinsic dimension using a bayesian formulation of 2nn.

        Args:
            alpha (float): parameter of the Gamma prior
            beta (float): parameter of the Gamma prior
            posterior_mean (bool): whether to use the posterior mean as estimator,
                if False the posterior mode will be used

        Returns:
            id (float): the estimated intrinsic dimension
            id_err (float): the standard error on the id estimation
            rs (float): the average nearest neighbor distance (rs)
        """
        if self.distances is None:
            self.compute_distances()

        if self.verb:
            print(
                "ID estimation started, using alpha = {} and beta = {}".format(
                    alpha, alpha
                )
            )

        distances_used = self.distances
        sum_log_mus = np.sum(np.log(distances_used[:, 2] / distances_used[:, 1]))

        alpha_post = alpha + self.N
        beta_post = beta + sum_log_mus

        mean_post = alpha_post / beta_post
        std_post = np.sqrt(alpha_post / beta_post**2)
        mode_post = (alpha_post - 1) / beta_post

        if posterior_mean:
            self.intrinsic_dim = mean_post
        else:
            self.intrinsic_dim = mode_post

        self.intrinsic_dim_err = std_post
        self.intrinsic_dim_scale = np.mean(distances_used[:, np.array([1, 2])])

        return self.intrinsic_dim, self.intrinsic_dim_err, self.intrinsic_dim_scale

    # ----------------------------------------------------------------------------------------------

    def _fix_rk(self, rk, r):
        """Compute the k_binomial points within the given rk and n_binomial points within given rn=rk*r.

        For each point, computes the number k_binomial of points within a sphere of radius rk
        and the number n_binomial within an inner sphere of radius rn=rk*r. It also provides
        a mask to take into account those points for which the statistics might be wrong, i.e.
        if k_binomial == self.maxk, there might be other points within rk that have not been taken into account
        because maxk was too low. If self.maxk is equal to the number of points of the dataset no mask will be applied

        Args:
            rk (float): external shell radius
            r (float): ratio between internal and external shell radii of the shells

        Returns:
            k (np.ndarray(int)): number of points within the external shell of radius rk
            n (np.ndarray(int)): number of points within the internal shell of radius rk*r
            mask (np.ndarray(bool)): array that states whether to use the point for the id estimate
        """
        # checks-in and initialisations
        if self.distances is None:
            self.compute_distances()

        assert rk > 0, "use a positive radius"
        assert 0 < r < 1, "select a proper ratio, 0<r<1"

        # routine
        rn = rk * r
        k = (self.distances <= rk).sum(axis=1)
        n = (self.distances <= rn).sum(axis=1)

        # checks-out
        if self.maxk == self.N - 1:
            mask = np.ones(self.N, dtype=bool)
        else:
            # if not all available NN were taken into account (i.e. maxk < N) and k is equal to self.maxk
            # or distances[:,-1]<lk, it is likely that there are other points within lk that are not being
            # considered and thus potentially altering the statistics -> neglect them through mask
            # in the calculation of likelihood
            mask = self.distances[:, -1] > rk  # or k == self.maxk

            if np.any(~mask):
                print(
                    "NB: for "
                    + str(sum(~(mask)))
                    + " points, the counting of k_binomial could be wrong, "
                    + "as more points might be present within the selected radius with respect "
                    "to the calculated neighbours. In order not to affect "
                    + "the statistics a mask is provided to remove them from the calculation of the "
                    + "likelihood or posterior.\nConsider recomputing NN with higher maxk or lowering Rk."
                )

        return k, n, mask

    # ----------------------------------------------------------------------------------------------

    def compute_id_binomial_rk(self, rk, r, bayes=True):
        """Calculate the id using the binomial estimator by fixing the same eternal radius for all the points.

        In the estimation of the id one has to remove the central point from the counting of n and k
        as it is not effectively part of the poisson process generating its neighbourhood.

        Args:
            rk (float): radius of the external shell
            r (float): ratio between internal and external shell
            bayes (bool, default=True): choose method between bayes (True) and mle (False). The bayesian estimate
                gives the mean value and std of d, while mle returns the max of the likelihood and the std
                according to Cramer-Rao lower bound

        Returns:
            id (float): the estimated intrinsic dimension
            id_err (float): the standard error on the id estimation
            rs (float): the average nearest neighbor distance (rs)

        """
        # checks-in and initialisations
        assert rk > 0, "Use a positive radius"
        assert 0 < r < 1, "Select a proper ratio, 0<r<1"

        # routine
        k, n, mask = self._fix_rk(rk, r)

        self.intrinsic_dim_scale = 0.5 * (rk + rk * r)

        n_eff = n[mask]
        k_eff = k[mask]

        e_n = n_eff.mean()
        e_k = k_eff.mean()
        if e_n == 1.0:
            print(
                "No points in the inner shell, returning 0. Consider increasing rk and/or the ratio"
            )
            self.intrinsic_dim = 0
            self.intrinsic_dim_err = 0
            return 0

        if bayes is False:
            self.intrinsic_dim = np.log((e_n - 1.0) / (e_k - 1.0)) / np.log(r)
            self.intrinsic_dim_err = np.sqrt(
                ut._compute_binomial_cramerrao(
                    self.intrinsic_dim, e_k - 1.0, r, n_eff.shape[0]
                )
            )

        elif bayes is True:
            (
                self.intrinsic_dim,
                self.intrinsic_dim_err,
                posterior_domain,
                posterior_values,
            ) = ut._beta_prior(k_eff - 1, n_eff - 1, r, posterior_profile=False)
        else:
            print("Select a proper method for id computation")
            return 0

        return self.intrinsic_dim, self.intrinsic_dim_err, self.intrinsic_dim_scale

    # ----------------------------------------------------------------------------------------------

    def _fix_k(self, k, r):
        """Compute rk, rn and n_binomial for each point of the dataset given a value of k.

        This routine computes the external radius rk, internal radius rn and internal points n
        given a value k, the number of NN to consider.

        Args:
            k (int): the number of NN to take into account
            r (float): ratio among rn and rk

        Returns:
            n (np.ndarray(int)): number of points within the internal shell of radius rk*r
        """
        # checks-in and initialisations
        # checks-in and initialisations
        if self.distances is None:
            self.compute_distances()

        assert (
            0 < k < self.maxk
        ), "Select a proper number of neighbours. Increase maxk and recompute distances if necessary"
        assert 0 < r < 1, "Select a proper ratio, 0<r<1"

        # routine
        rk = self.distances[:, k]
        rn = rk * r
        n = (self.distances <= rn.reshape(self.N, 1)).sum(axis=1)

        self.intrinsic_dim_scale = 0.5 * (rk.mean() + rn.mean())

        return n

    # --------------------------------------------------------------------------------------

    def compute_id_binomial_k(self, k, r, bayes=True):
        """Calculate id using the binomial estimator by fixing the number of neighbours.

        As in the case in which one fixes rk, also in this version of the estimation
        one removes the central point from n and k. Furthermore, one has to remove also
        the k-th NN, as it plays the role of the distance at which rk is taken.
        So if k=5 it means the 5th NN from the central point will be considered,
        taking into account 6 points though (the central one too). This means that
        in principle k_eff = 6, to which I'm supposed to subtract 2. For this reason
        in the computation of the MLE we have directly k-1, which explicitly would be k_eff-2

        Args:
            k (int): number of neighbours to take into account
            r (float): ratio between internal and external shells
            bayes (bool, default=True): choose method between bayes (True) and mle (False). The bayesian estimate
                gives the mean value and std of d, while mle returns the max of the likelihood and the std
                according to Cramer-Rao lower bound

        Returns:
            id (float): the estimated intrinsic dimension
            id_err (float): the standard error on the id estimation
            rs (float): the average nearest neighbor distance (rs)
        """
        # checks-in and initialisations
        assert (
            0 < k < self.maxk
        ), "Select a proper number of neighbours. Increase maxk if necessary"
        assert 0 < r < 1, "Select a proper ratio, 0<r<1"

        # routine
        n = self._fix_k(k, r)
        e_n = n.mean()
        if e_n == 1.0:
            print(
                "no points in the inner shell, returning 0\n. Consider increasing rk and/or the ratio"
            )
            self.intrinsic_dim = 0
            self.intrinsic_dim_err = 0
            return 0

        if bayes is False:
            self.intrinsic_dim = np.log((e_n - 1) / (k - 1)) / np.log(r)
            self.intrinsic_dim_err = np.sqrt(
                ut._compute_binomial_cramerrao(self.intrinsic_dim, k - 1, r, n.shape[0])
            )

        elif bayes is True:
            (
                self.intrinsic_dim,
                self.intrinsic_dim_err,
                posterior_domain,
                posterior_values,
            ) = ut._beta_prior(k - 1, n - 1, r, posterior_profile=False)
        else:
            print("select a proper method for id computation")
            return 0

        return self.intrinsic_dim, self.intrinsic_dim_err, self.intrinsic_dim_scale

    # ----------------------------------------------------------------------------------------------
    def set_id(self, d):
        """Set the intrinsic dimension."""
        assert d > 0, "intrinsic dimension can't be negative (yet)"
        self.intrinsic_dim = d
