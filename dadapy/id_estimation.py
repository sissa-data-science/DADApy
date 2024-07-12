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
from dadapy._utils.id_estimation import _binomial_model_validation as bmv
from dadapy._utils.utils import compute_nn_distances
from dadapy.base import Base

cores = multiprocessing.cpu_count()


class IdEstimation(Base):
    """IdEstimation class."""

    def __init__(self, *args, **kwargs):
        """Estimate the intrinsic dimension of a dataset choosing among various routines.

        Inherits from class Base.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            distances (np.ndarray(float)): A matrix of dimension N x mask containing distances between points
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            period (np.array(float), optional): array containing periodicity for each coordinate. Default is None
            verbose (bool): whether you want the code to speak or shut up
            n_jobs (int): number of cores to be used
        """
        self.intrinsic_dim = None
        self.intrinsic_dim_err = None
        self.intrinsic_dim_scale = None
        self.intrinsic_dim_mus = None
        self.intrinsic_dim_mus_gride = None

        super().__init__(*args, **kwargs)
        if self.n_jobs is None:
            self.n_jobs = cores

    # ----------------------------------------------------------------------------------------------

    def _compute_id_2NN(self, mus, mu_fraction, algorithm="base"):
        """Compute the id using the 2NN algorithm.

        Helper of return return_id_2NN.

        Args:
            mus (np.ndarray(float)): ratio of the distances of first- and second-nearest neighbours
            mu_fraction (float): fraction of mus to take into account, discard the highest values
            algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood

        Returns:
            intrinsic_dim (float): the estimation of the intrinsic dimension
        """
        N = mus.shape[0]
        n_eff = int(N * mu_fraction)
        log_mus = np.log(mus)
        log_mus_reduced = np.sort(log_mus)[:n_eff]

        if algorithm == "ml":
            intrinsic_dim = (N - 1) / np.sum(log_mus)

        elif algorithm == "base":
            y = -np.log(1 - np.arange(1, n_eff + 1) / N)

            def func(x, m):
                return m * x

            intrinsic_dim, _ = curve_fit(func, log_mus_reduced, y)
            # curve_fit returns a 1-element array
            intrinsic_dim = intrinsic_dim[0]

        else:
            raise ValueError("Please select a valid algorithm type")

        return intrinsic_dim

    # ----------------------------------------------------------------------------------------------
    def compute_id_2NN(
        self,
        algorithm="base",
        mu_fraction=0.9,
        data_fraction=1,
        n_iter=None,
        set_attr=True,
    ):
        """Compute intrinsic dimension using the 2NN algorithm.

        Args:
            algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood
            mu_fraction (float): fraction of mus that will be considered for the estimate (discard highest mus)
            data_fraction (float): fraction of randomly sampled points used to compute the id
            n_iter (int): number of times the ID is computed on data subsets (useful when decimation < 1)
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
        assert (self.X is not None) or (
            self.distances is not None
        ), """2NN algorithm requires that either self.X or self.distances is not None.\
            Please initialize a coordinate or distance matrix."""
        assert (
            0.0 < data_fraction and data_fraction <= 1.0
        ), "'data_fraction' must be between 0 and 1"
        assert (
            0.0 < mu_fraction and mu_fraction <= 1.0
        ), "'fraction' must be between 0 and 1"
        if math.isclose(mu_fraction, 1.0) and algorithm == "base":
            algorithm = "ml"
            print("fraction = 1: algorithm set to ml")

        if data_fraction == 1:
            if self.distances is None:
                self.compute_distances()

            mus = self.distances[:, 2] / self.distances[:, 1]
            intrinsic_dim = self._compute_id_2NN(mus, mu_fraction, algorithm)
            intrinsic_dim_err = 0.0
            intrinsic_dim_scale = np.mean(self.distances[:, np.array([1, 2])])

        else:
            n_subset = int(np.rint(self.N * data_fraction))
            if n_iter is None:
                n_iter = int(np.rint(1.0 / data_fraction))

            mus, id_list, r_list = self._compute_id_iterated(
                n_iter, n_subset, mu_fraction, algorithm
            )

            intrinsic_dim = np.mean(id_list)
            intrinsic_dim_err = np.std(id_list) / len(id_list) ** 0.5
            intrinsic_dim_scale = np.mean(r_list)

        if self.verb:
            print(f"ID estimation finished: selecting ID of {intrinsic_dim}")

        if set_attr:
            self.intrinsic_dim = intrinsic_dim
            self.intrinsic_dim_err = intrinsic_dim_err
            self.intrinsic_dim_scale = intrinsic_dim_scale
            self.intrinsic_dim_mus = mus

        return intrinsic_dim, intrinsic_dim_err, intrinsic_dim_scale

    # ----------------------------------------------------------------------------------------------

    def _compute_id_iterated(self, n_iter, n_subset, fraction, algorithm):
        ids = np.zeros(n_iter)
        rs = np.zeros(n_iter)
        n_survived = np.zeros(n_iter)
        mus = [[] for _ in range(n_iter)]
        decimation_from_distances = self.X is None and self.distances is not None

        for j in range(n_iter):
            # do decimation from pure distance matrix
            if decimation_from_distances:
                # random subsample a subset of indices
                indices = self.rng.choice(self.N, n_subset, replace=False)
                # Is self.dist_indices[i, j] selected?
                mask = np.isin(self.dist_indices, indices)

                # distance matrix where the selected indices also have two nearest neighbors within self.maxk
                distances = []
                for index in indices:
                    if np.sum(mask[index]) > 2:
                        distances.append(self.distances[index, mask[index]][:3])
                distances = np.array(distances)
                n_survived[j] = distances.shape[0]

            else:
                idx = self.rng.choice(self.N, size=n_subset, replace=False)
                x_decimated = self.X[idx]
                distances, _ = compute_nn_distances(
                    x_decimated,
                    maxk=3,  # only compute first 2 nn
                    metric=self.metric,
                    period=self.period,
                    n_jobs=self.n_jobs,
                )

            mus[j] = distances[:, 2] / distances[:, 1]
            ids[j] = self._compute_id_2NN(mus[j], fraction, algorithm)
            rs[j] = np.mean(distances[:, np.array([1, 2])])

        if decimation_from_distances and (np.mean(n_survived) < 0.8 * n_subset):
            warnings.warn(
                f"""Decimation from a sparse distance matrix uses
                on average {int(np.mean(n_survived))} out of the {n_subset} data points. """,
                stacklevel=2,
            )

        return mus, ids, rs

    def return_id_scaling_2NN(
        self,
        n_min=10,
        algorithm="base",
        mu_fraction=0.9,
        set_attr=False,
        return_sizes=False,
    ):
        """Compute the id with the 2NN algorithm at different scales.

        The different scales are obtained by sampling subsets of [N, N/2, N/4, N/8, ..., n_min] data points.

        Args:
            n_min (int): minimum number of points considered when decimating the dataset,
                        n_min effectively sets the largest 'scale';
            algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood;
            mu_fraction (float): fraction of mus that will be considered for the estimate (discard highest mus).

        Returns:
            ids_scaling (np.ndarray(float)): array of intrinsic dimensions;
            ids_scaling_err (np.ndarray(float)): array of error estimates;
            scales (np.ndarray(int)): array of maximum nearest neighbor rank included in the estimate

        Quick Start:
        ===========

        .. code-block:: python

                from dadapy import Data
                from sklearn.datasets import make_swiss_roll

                #two dimensional curved manifold embedded in 3d with noise

                n_samples = 5000
                X, _ = make_swiss_roll(n_samples, noise=0.3)

                ie = Data(coordinates=X)
                ids_scaling, ids_scaling_err, rs_scaling = ie.return_id_scaling_2NN(n_min = 20)

                ids_scaling:
                array([2.88 2.77 2.65 2.42 2.22 2.2  2.1  2.23])

                ids_scaling_err:
                array([0.   0.02 0.05 0.04 0.04 0.03 0.04 0.04])

                scales:
                array([2  4  8  16  32  64  128  256])
        """
        max_ndec = int(math.log(self.N, 2)) - 1
        num_subsets = np.round(self.N / np.array([2**i for i in range(max_ndec)]))
        num_subsets = num_subsets.astype(int)

        if n_min is not None:
            num_subsets = num_subsets[num_subsets > n_min]

        ids_scaling = np.zeros(num_subsets.shape[0])
        ids_scaling_err = np.zeros(num_subsets.shape[0])
        rs_scaling = np.zeros((num_subsets.shape[0]))

        mus = []
        for i, num_subset in enumerate(num_subsets):
            ids_scaling[i], ids_scaling_err[i], rs_scaling[i] = self.compute_id_2NN(
                algorithm=algorithm,
                mu_fraction=mu_fraction,
                data_fraction=num_subset / self.N,
                set_attr=True,
            )
            mus.append(self.intrinsic_dim_mus)

        if set_attr:
            self.intrinsic_dim_decimation = ids_scaling
            self.intrinsic_dim_err_decimation = ids_scaling_err
            self.intrinsic_dim_scale_decimation = rs_scaling
            self.intrinsic_dim_mus_decimation = mus

        scales = rs_scaling
        if return_sizes:
            scales = num_subsets

        return ids_scaling, ids_scaling_err, scales

    # ----------------------------------------------------------------------------------------------
    def return_id_scaling_gride(
        self,
        range_max=64,
        d0=0.001,
        d1=1000,
        eps=1e-7,
        set_attr=False,
        return_ranks=False,
    ):
        """Compute the id at different scales using the Gride algorithm.

        Args:
            range_max (int): maximum nearest neighbor rank considered for the id computations;
                            the number of id estimates are log2(range_max) as the nearest neighbor
                            order ('scale') is doubled at each estimate;
            d0 (float): minimum intrinsic dimension considered in the search;
            d1 (float): maximum intrinsic dimension considered in the search;
            eps (float): precision of the approximate id calculation.
            set_attr (bool): whether to change the class attributes as a result of the computation

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
        assert (self.X is not None) or (
            self.distances is not None
        ), """2NN algorithm requires that either self.X or self.distances is not None.\
            Please initialize a coordinate or distance matrix."""

        max_rank = min(self.N, range_max)
        if self.X is None and max_rank > self.maxk:
            max_rank = self.maxk
            warnings.warn(
                f"""{range_max} set to {range_max} but data class initialized with\
                    a sparse distance matrix with only {self.maxk} nearest neighbors.\
                    {range_max} is set to {self.maxk}""",
                stacklevel=2,
            )
        max_step = int(math.log(max_rank, 2))
        nn_ranks = np.array([2**i for i in range(max_step + 1)])

        if self.distances is not None and max_rank < self.maxk + 1:
            mus = self.distances[:, nn_ranks[1:]] / self.distances[:, nn_ranks[:-1]]
            rs = self.distances[:, np.array([nn_ranks[:-1], nn_ranks[1:]])]

        elif self.X is not None:
            distances, dist_indices, mus, rs = self._return_mus_scaling(
                range_scaling=max_rank
            )

            if self.verb:
                print("distance computation finished")

            # if distances have not been computed save them
            if self.distances is None:
                self.distances = distances
                self.dist_indices = dist_indices
                self.N = distances.shape[0]

        # compute IDs (and their error) via maximum likelihood for all the scales up to max_rank
        ids_scaling, ids_scaling_err = self._compute_id_gride_multiscale(
            mus, d0, d1, eps
        )
        if self.verb:
            print("id computation finished")

        "average of the kth and 2*kth neighbor distances taken over all datapoints for each id estimate"
        rs_scaling = np.mean(rs, axis=(0, 1))

        if set_attr:
            self.intrinsic_dim_gride = ids_scaling
            self.intrinsic_dim_err_gride = ids_scaling_err
            self.intrinsic_dim_scale_gride = rs_scaling
            self.intrinsic_dim_mus_gride = mus

        scales = rs_scaling
        if return_ranks:
            scales = nn_ranks[1:]

        return ids_scaling, ids_scaling_err, scales

    # ----------------------------------------------------------------------------------------------
    def _compute_id_gride_multiscale(self, mus, d0, d1, eps):
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

            intrinsic_dim, id_error = self._compute_id_gride_single_scale(
                d0, d1, mus[:, i], n1, 2 * n1, eps
            )

            ids_scaling[i] = intrinsic_dim
            ids_scaling_err[i] = id_error

        return ids_scaling, ids_scaling_err

    def _compute_id_gride_single_scale(self, d0, d1, mus, n1, n2, eps):
        id_ = ut._argmax_loglik(
            self.dtype, d0, d1, mus, n1, n2, eps=eps
        )  # eps=precision id calculation
        id_err = (
            1
            / ut._fisher_info_scaling(
                id_, mus, n1, 2 * n1, eps=5 * self.eps
            )  # eps=regularization small numbers
        ) ** 0.5

        return id_, id_err

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
        steps = np.array([2**i for i in range(max_step + 1)])

        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, steps[-1], axis=1)
        neigh_ind = neigh_ind[:, : steps[-1] + 1]

        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]

        dist = np.sqrt(dist[sample_range, neigh_ind])
        dist = self.remove_zero_dists(dist)
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
                n_jobs=self.n_jobs,
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
                this may compromise the correct behavior of some routines""",
                stacklevel=2,
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
            rk (float or np.ndarray(float)): external shell radius
            r (float): ratio between internal and external shell radii of the shells

        Returns:
            k (np.ndarray(int)): number of points within the external shell of radius rk
            n (np.ndarray(int)): number of points within the internal shell of radius rk*r
            mask (np.ndarray(bool)): array that states whether to use the point for the id estimate
        """
        # checks-in and initialisations
        if self.distances is None:
            self.compute_distances()

        assert 0 < r < 1, "Select a proper ratio, 0<r<1"

        if isinstance(rk, np.ndarray):
            assert np.all(rk > 0), "Not all radii are positive"
            assert (
                rk.shape[0] == self.N
            ), "array of radii must have the same length of datapoints"
            rn = rk * r
            k = np.sum([d < ri for d, ri in zip(self.distances, rk)], axis=1)
            n = np.sum([d < ri for d, ri in zip(self.distances, rn)], axis=1)
        else:
            assert rk > 0, "Use a positive radius"
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

    def compute_id_binomial_rk(
        self, rk, r, bayes=True, plot_mv=False, plot_posterior=False
    ):
        """Calculate the id using the binomial estimator by fixing the same eternal radius for all the points.

        In the estimation of the id one has to remove the central point from the counting of n and k
        as it is not effectively part of the poisson process generating its neighbourhood.

        Args:
            rk (float or np.ndarray(float)): radius of the external shell
            r (float): ratio between internal and external shell
            bayes (bool, default=True): choose method between bayes (True) and mle (False). The bayesian estimate
                gives the mean value and std of d, while mle returns the max of the likelihood and the std
                according to Cramer-Rao lower bound
            plot_mv (bool, default=False): whether to print the output of the model validation
            plot_posterior (bool, default=False): if True, together with bayes, plots the posterior of the ID

        Returns:
            id (float): the estimated intrinsic dimension
            id_err (float): the standard error on the id estimation
            scale (float): scale at which the id is performed
            pv (float): p-value of the test statistics computed with Epps-Singleton model validation

        """
        k, n, mask = self._fix_rk(rk, r)

        self.intrinsic_dim_scale = (
            0.5 * (rk.mean() + (rk * r).mean())
            if isinstance(rk, np.ndarray)
            else 0.5 * (rk + rk * r)
        )

        n_eff = n[mask]
        k_eff = k[mask]

        e_n = n_eff.mean()
        e_k = k_eff.mean()
        if math.isclose(e_n, 1.0):
            print(
                "No points in the inner shell, returning 0. Consider increasing rk and/or the ratio"
            )
            self.intrinsic_dim = 0
            self.intrinsic_dim_err = 0
            return 0

        if not bayes:
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
            ) = ut._beta_prior(
                k_eff - 1.0, n_eff - 1.0, r, posterior_profile=plot_posterior
            )
        else:
            print("Select a proper method for id computation")
            return 0

        ks, pv = bmv(k_eff, n_eff, r**self.intrinsic_dim, plot=plot_mv)

        return self.intrinsic_dim, self.intrinsic_dim_err, self.intrinsic_dim_scale, pv

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
        if self.distances is None:
            self.compute_distances()

        assert 0 < r < 1, "Select a proper ratio, 0<r<1"

        if isinstance(k, np.ndarray):
            assert np.all(k > 0), "Not all ks are positive"
            assert np.all(k <= self.maxk), "Some ks are larger than maxk"
            assert (
                k.shape[0] == self.N
            ), "array of ks must have the same length of datapoints"
            rk = np.array([di[ki] for di, ki in zip(self.distances, k)])
            rn = rk * r
            n = np.sum([di < ri for di, ri in zip(self.distances, rn)], axis=1)

        else:
            assert (
                0 < k < self.maxk
            ), "Select a proper number of neighbours. Increase maxk and recompute distances if necessary"
            rk = self.distances[:, k]
            rn = rk * r
            n = (self.distances <= rn.reshape(self.N, 1)).sum(axis=1)

        self.intrinsic_dim_scale = 0.5 * (rk.mean() + rn.mean())

        return n

    # --------------------------------------------------------------------------------------

    def compute_id_binomial_k(
        self, k, r, bayes=True, plot_mv=False, plot_posterior=False, k_bootstrap=1
    ):
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
            plot_mv (bool, default=False): whether to print the output of the model validation
            plot_posterior (bool, default=False): if True, together with bayes, plots the posterior of the ID

        Returns:
            id (float): the estimated intrinsic dimension
            id_err (float): the standard error on the id estimation
            scale (float): the average nearest neighbor distance (rs)
            pv (float): p-value of the test statistics through Epps-Singleton test
        """
        n = self._fix_k(k, r)
        e_n = n.mean()
        if math.isclose(e_n, 1.0):
            print(
                "no points in the inner shell, returning 0\n. Consider increasing k and/or the ratio"
            )
            self.intrinsic_dim = 0
            self.intrinsic_dim_err = 0
            return 0

        k_eff = k.mean() if isinstance(k, np.ndarray) else k

        if bayes is False:
            self.intrinsic_dim = np.log((e_n - 1.0) / (k_eff - 1.0)) / np.log(r)
            self.intrinsic_dim_err = np.sqrt(
                ut._compute_binomial_cramerrao(
                    self.intrinsic_dim, k_eff - 1, r, n.shape[0]
                )
            )

        elif bayes is True:
            (
                self.intrinsic_dim,
                self.intrinsic_dim_err,
                posterior_domain,
                posterior_values,
            ) = ut._beta_prior(
                k_eff - 1.0, n - 1.0, r, posterior_profile=plot_posterior
            )
        else:
            print("select a proper method for id computation")
            return 0

        ks, pv = bmv(
            k, n, r**self.intrinsic_dim, plot=plot_mv, k_bootstrap=k_bootstrap
        )

        return self.intrinsic_dim, self.intrinsic_dim_err, self.intrinsic_dim_scale, pv

    # ----------------------------------------------------------------------------------------------

    def set_id(self, d):
        """Set the intrinsic dimension."""
        assert d > 0, "intrinsic dimension can't be negative (yet)"
        self.intrinsic_dim = d
