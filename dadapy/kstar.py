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

import multiprocessing
import time
import warnings

import numpy as np

from dadapy._cython import cython_density as cd
from dadapy.id_estimation import IdEstimation

cores = multiprocessing.cpu_count()


class KStar(IdEstimation):
    """Computes for each point an optimal choice - kstar - of the neighbourhood size.

    Inherits from class IdEstimation.
    Can assign to the data a user-defined neighbourhood size.

    Attributes:
        kstar (np.array(float)): array containing the chosen number k* in the neighbourhood of each of the N points
        dc (np.array(float), optional): array containing the distance of the k*th neighbor from each of the N points
    """

    def __init__(
        self, coordinates=None, distances=None, maxk=None, verbose=False, n_jobs=cores
    ):
        """Initialise the KStar class."""
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            n_jobs=n_jobs,
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
                "Setting the k value but, be careful: the intrinsic dimension is not defined!"
            )

        if isinstance(k, np.ndarray):
            self.kstar = k
        else:
            self.kstar = np.full(self.N, k, dtype=int)

    # ----------------------------------------------------------------------------------------------

    def compute_kstar(self, Dthr=23.92812698):
        """Compute an optimal choice of the neighbourhood size k for each point.

        Args:
            Dthr (float): Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
                to a p-value of 1e-6.

        """
        if self.intrinsic_dim is None:
            warnings.warn(
                "Careful! The intrinsic dimension is not defined. "
                "Computing it unsupervisedly with 'compute_id_2NN()' method"
            )
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
