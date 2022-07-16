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
The *density_estimation* module contains the *DensityEstimation* class.

The different algorithms of density estimation are implemented as methods of this class.
"""

import multiprocessing
import time

import numpy as np

from dadapy._cython import cython_density as cd
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

    Attributes:
        kstar (np.array(float)): array containing the chosen number k* in the neighbourhood of each of the N points
        dc (np.array(float), optional): array containing the distance of the k*th neighbor from each of the N points
        log_den (np.array(float), optional): array containing the N log-densities
        log_den_err (np.array(float), optional): array containing the N errors on the log_den

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
        self.dc = None
        self.log_den = None
        self.log_den_err = None

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

        self.dc = None
        self.log_den = None
        self.log_den_err = None

    # ----------------------------------------------------------------------------------------------

    def compute_density_kNN(self, k=10):
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
            self.distances, self.intrinsic_dim, self.kstar, interpolation=False
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

    def compute_density_kstarNN(self, Dthr=23.92812698):
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
            self.distances, self.intrinsic_dim, self.kstar, interpolation=False
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

        dc = np.zeros(self.N, dtype=float)
        log_den = np.zeros(self.N, dtype=float)
        log_den_err = np.zeros(self.N, dtype=float)
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

    def compute_density_PAk(self, Dthr=23.92812698, optimized=True):
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
            )

        else:
            log_den, log_den_err, dc = return_not_normalised_density_PAk(
                self.distances,
                self.intrinsic_dim,
                self.kstar,
                self.maxk,
                interpolation=False,
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
