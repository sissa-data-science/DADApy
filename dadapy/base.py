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
The *base* module contains the *Base* class.

This class contains essential methods and attributes needed for all other classes.
"""
import multiprocessing
import time
import warnings

import numpy as np

from dadapy._utils.utils import compute_nn_distances, from_all_distances_to_nndistances

cores = multiprocessing.cpu_count()


class Base:
    """Base class."""

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
        """Containing coordinates and/or distances and some basic methods.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            distances (np.ndarray(float), tuple(np.ndarray(float), np.ndarray(float)) ): Distance matrix (N x N),
                                        or tuple of nearest neighbor distances (N x maxk) and their indices (N x maxk).
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            period (np.array(float), optional): array containing the periodicity of each coordinate. Default is None
            verbose (bool): whether you want the code to speak or shut up
            njobs (int): number of cores to be used
        """
        self.X = coordinates
        self.maxk = maxk  # remove from here
        self.verb = verbose
        self.n_jobs = n_jobs
        self.dims = None
        self.N = None
        self.metric = "euclidean"  # remove from here
        self.period = period  # remove from here
        self.rng = np.random.default_rng(rng_seed)

        if self.X is not None:
            assert isinstance(
                self.X, np.ndarray
            ), "Coordinates must be in numpy ndarray format"
            if (
                self.X.dtype == np.float32
                or self.X.dtype == np.float16
                or self.X.dtype == np.float64
            ):
                self.X = self.X.astype(np.float64, casting="safe")
            else:
                warnings.warn(
                    f"data type is {self.X.dtype}: most methods work only with float-type inputs",
                    stacklevel=2,
                )

            self.N = self.X.shape[0]
            self.dims = coordinates.shape[1]
            self.distances = None
            self.dist_indices = None
            if self.maxk is None:  # remove from here
                self.maxk = min(100, self.N - 1)

        if distances is not None:
            self.distances, self.dist_indices, self.N, self.maxk = self._init_distances(
                distances, self.maxk
            )

        self.dtype = np.float64
        self.eps = np.finfo(self.dtype).eps

    # this function is useful for overlap with another dataset and makes __init__ little bit shorter
    def _init_distances(self, distances, maxk=None):
        if isinstance(distances, tuple):
            assert isinstance(
                distances[0], np.ndarray
            ), "distances must be in numpy ndarray format"
            assert isinstance(
                distances[1], np.ndarray
            ), "distance indices must be in numpy ndarray format"

            assert (
                distances[0].shape[0] == distances[1].shape[0]
            ), "distances and indices must have the same shape"

            if maxk is None:
                maxk = min(100, distances[0].shape[1] - 1)

            elif maxk > (distances[0].shape[1] - 1):
                maxk = distances[0].shape[1] - 1
                warnings.warn(
                    f"maxk requested bigger than number of features: setting maxk to {maxk}",
                    stacklevel=2,
                )

            N = distances[0].shape[0]

            dist = distances[0][:, : maxk + 1]
            dist_indices = distances[1][:, : maxk + 1]

        else:
            assert isinstance(
                distances, np.ndarray
            ), "distances must be in numpy ndarray format"
            assert (
                distances.shape[0] == distances.shape[1]
            ), "distance matrix shape must be N x N"

            N = distances.shape[0]
            if maxk is None:
                maxk = min(100, distances.shape[1] - 1)

            elif maxk > (distances.shape[1] - 1):
                maxk = distances.shape[1] - 1
                warnings.warn(
                    f"maxk requested bigger than number of features: setting maxk to {maxk}",
                    stacklevel=2,
                )

            dist, dist_indices = from_all_distances_to_nndistances(distances, maxk)

        if dist.dtype == np.float32:
            dist = dist.astype(np.float64, casting="safe")

        return dist, dist_indices, N, maxk

    # ----------------------------------------------------------------------------------------------

    def compute_distances(
        self, maxk=None, metric="euclidean", period=None, n_jobs=None
    ):
        """Compute distaces between points up to the maxk nearest neighbour.

        Args:
            maxk: maximum number of neighbours for which distance is computed and stored
            metric: type of metric
            period (float or np.array): periodicity (only used for periodic distance computation). Default is None.

        """
        if self.verb:
            print("Computation of distances started")
            sec = time.time()

        self.metric = metric

        if period is not None:
            if isinstance(period, np.ndarray) and period.shape == (self.dims,):
                self.period = period
            elif isinstance(period, (int, float)):
                self.period = np.full((self.dims), fill_value=period, dtype=float)
            else:
                raise ValueError(
                    f"'period' must be either a float scalar or a numpy array of floats of shape ({self.dims},)"
                )

        if maxk is not None:
            self.maxk = maxk
        else:
            assert (
                self.maxk is not None
            ), "set parameter maxk in the function or for the class"

        if self.verb and period is not None:
            print(
                "Computing periodic distances.",
                "The coordinates are assumed to be in the range [0, period]",
            )

        if n_jobs is not None:
            self.n_jobs = n_jobs

        if self.verb:
            print(f"Computation of the distances up to {self.maxk} NNs started")

        self.distances, self.dist_indices = compute_nn_distances(
            self.X, self.maxk, self.metric, self.period, self.n_jobs
        )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds for computing distances".format(sec2 - sec))

    # -------------------------------------------------------------------------------

    # better to use this formulation which can be applied to _mus_scaling_reduce_func
    def remove_zero_dists(self, distances):
        """Find zero neighbour distances and substitute the numerical zeros with a very small number.

        This method is mostly useful to regularize the computation of certain id estimators.
        """
        # find all distances which are 0
        indices = np.nonzero(distances[:, 1:] < np.finfo(self.dtype).eps)
        # set distance to epsilon
        distances[:, 1:][indices] = np.finfo(self.dtype).eps

        return distances

    def remove_identical_points(self):
        """Find points that are numerically identical and remove them.

        For very large datasets this method might be slow; you might want to use a command like: awk '!seen[$0]++' .
        See
        https://unix.stackexchange.com/questions/11939/how-to-get-only-the-unique-results-without-having-to-sort-data
        for more information
        """
        if self.N > 100000:
            print(
                "WARNING: this method might be very slow for large datasets.\n"
                "We suggest to use something like \"awk '!seen[$0]++'\""
            )

        # removal of overlapping data points
        x_unique = np.unique(self.X, axis=0)

        n_unique = x_unique.shape[0]

        if n_unique < self.N:
            print(
                f"{self.N - n_unique} overlapping datapoints found: keeping {n_unique} unique elements",
                "WARNING: the order of points has been changed!",
            )

            self.X = x_unique
            self.N = n_unique
            self.maxk = min(self.maxk, self.N - 1)

            if self.distances is not None:
                print("distances between points will be recomputed")
                self.compute_distances()

        else:
            print("No identical identical points were found")
