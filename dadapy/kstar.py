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
from scipy.stats import chi2
from scipy.special import gammaln
import math
from tqdm import tqdm

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

    def compute_kstar(self, alpha=1e-6, bonferroni_deloc=False, bonferroni_loc=False):
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
            bonferroni_loc
        )

        self.set_kstar(kstar)

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing kstar".format(sec2 - sec))

    def compute_kstar_python(self, alpha, bonferroni_deloc=False, bonferroni_loc=False):
        """Pure Python version of _compute_kstar from cython_density.pyx."""
        Nele=self.N
        id_sel = self.intrinsic_dim
        maxk=self.maxk
        kstar = np.empty(Nele, dtype=int)
        prefactor = math.exp(id_sel / 2.0 * math.log(math.pi) - gammaln((id_sel + 2.0) / 2.0))
        alpha_eff = alpha / Nele if bonferroni_deloc else alpha
        Dthr = chi2(1).isf(alpha_eff)

        for i in tqdm(range(Nele)):
            j = 4
            dL = 0.0
            h = 0
            dL_arr = np.empty(maxk, dtype=float)
            if bonferroni_loc:
                while j < maxk:
                    ksel = j - 1
                    vvi = prefactor * (self.distances[i, ksel] ** id_sel)
                    vvj = prefactor * (self.distances[self.dist_indices[i, j], ksel] ** id_sel)
                    dL = -2.0 * ksel * (math.log(vvi) + math.log(vvj) - 2.0 * math.log(vvi + vvj) + math.log(4))
                    dL_arr[h] = dL
                    h += 1
                    Dthr_loc = chi2(1).isf(alpha_eff / h)
                    all_pass = True
                    for k in range(h):
                        if dL_arr[k] > Dthr_loc:
                            all_pass = False
                            break
                    if not all_pass:
                        break
                    j += 1
                if j == 4:
                    kstar[i] = 3
                else:
                    kstar[i] = j - 2
            else:
                while j < maxk and dL < Dthr:
                    ksel = j - 1
                    vvi = prefactor * (self.distances[i, ksel] ** id_sel)
                    vvj = prefactor * (self.distances[self.dist_indices[i, j], ksel] ** id_sel)
                    dL = -2.0 * ksel * (math.log(vvi) + math.log(vvj) - 2.0 * math.log(vvi + vvj) + math.log(4))
                    j += 1
                if j == maxk:
                    kstar[i] = j - 1
                else:
                    kstar[i] = j - 2
        return kstar
