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
The *base* module contains the *Base* class.

This class contains essential methods and attributes needed for all other classes.
"""
import multiprocessing
import time

import numpy as np

from dadapy._utils.utils import compute_nn_distances, from_all_distances_to_nndistances

cores = multiprocessing.cpu_count()
rng = np.random.default_rng()


class Base:
    """Base class."""

    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        njobs=cores,
    ):
        """Containing coordinates and/or distances and some basic methods.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            distances (np.ndarray(float)): A matrix of dimension N x mask containing distances between points
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            period (np.array(float), optional): array containing the periodicity of each coordinate. Default is None
            verbose (bool): whether you want the code to speak or shut up
            njobs (int): number of cores to be used
        """
        self.X = coordinates
        self.maxk = maxk
        self.verb = verbose
        self.njobs = njobs
        self.dims = None
        self.N = None
        self.metric = "euclidean"
        self.period = period

        if coordinates is not None:
            assert isinstance(
                self.X, np.ndarray
            ), "Coordinates must be in numpy ndarray format"

            self.dtype = self.X.dtype

            self.N = self.X.shape[0]
            self.dims = coordinates.shape[1]
            self.distances = None
            if self.maxk is None:
                self.maxk = min(100, self.N - 1)

        if distances is not None:
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

                if self.maxk is None:
                    self.maxk = min(100, distances[0].shape[1] - 1)

                self.N = distances[0].shape[0]
                self.distances = distances[0][:, : self.maxk + 1]
                self.dist_indices = distances[1][:, : self.maxk + 1]

            else:
                assert isinstance(
                    distances, np.ndarray
                ), "distances must be in numpy ndarray format"
                assert (
                    distances.shape[0] == distances.shape[1]
                ), "distance matrix shape must be N x N"

                self.N = distances.shape[0]
                if self.maxk is None:
                    self.maxk = min(100, distances.shape[1] - 1)

                self.distances, self.dist_indices = from_all_distances_to_nndistances(
                    distances, self.maxk
                )

            self.dtype = self.distances.dtype
        try:
            self.eps = np.finfo(self.dtype).eps
        except BaseException:
            self.eps = None

    # ----------------------------------------------------------------------------------------------

    def compute_distances(self, maxk=None, metric="euclidean", period=None):
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

        if self.verb:
            print(f"Computation of the distances up to {self.maxk} NNs started")

        self.distances, self.dist_indices = compute_nn_distances(
            self.X, self.maxk, self.metric, self.period
        )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds for computing distances".format(sec2 - sec))

    # -------------------------------------------------------------------------------

    # better to use this formulation which can be applied to _mus_scaling_reduce_func
    def _remove_zero_dists(self, distances):
        """Find zero nearest neighhbour distances and substitute the numerical zeros with a very small number.

        This method is mostly useful to regularize the computation of certain id estimators.

        Args:
            distances: distances to modify

        Returns:
            distances with regularised zeros

        """
        # find all points with any zero distance
        indx_ = np.nonzero(distances[:, 1] < np.finfo(self.dtype).eps)[0]
        # set nearest distance to eps:
        distances[indx_, 1] = np.finfo(self.dtype).eps

        return distances

    def remove_identical_points(self):
        """Find points that are numerically identical and remove them.

        For very large datasets this method might be slow and you might want to use a command like: awk '!seen[$0]++' .
        See
        https://unix.stackexchange.com/questions/11939/how-to-get-only-the-unique-results-without-having-to-sort-data
        for more information
        """
        if self.N > 100000:
            print("WARNING: this method might be very slow for large datasets. ")

        # removal of overlapping data points
        X_unique = np.unique(self.X, axis=0)

        N_unique = X_unique.shape[0]

        if N_unique < self.N:

            print(
                f"{self.N - N_unique} overlapping datapoints found: keeping {N_unique} unique elements",
                "WARNING: the order of points has been changed!",
            )

            self.X = X_unique
            self.N = N_unique
            self.maxk = min(self.maxk, self.N - 1)

            if self.distances is not None:
                print("distances between points will be recomputed")
                self.compute_distances()

        else:
            print("No identical identical points were found")
