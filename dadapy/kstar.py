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
from scipy.special import gammaln

from dadapy._cython import cython_density as cd
from dadapy.id_estimation import IdEstimation

try:
    import jax.numpy as jnp

    _HAS_JAX = True
except ModuleNotFoundError:
    jnp = None
    _HAS_JAX = False

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
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        n_jobs=cores,
    ):
        """Initialise the KStar class."""
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
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

    def compute_kstar(
        self, Dthr=23.92812698,
        backend="cython",
        batch_size=None,
        n_jobs=None
    ):
        """Compute an optimal choice of the neighbourhood size k for each point.

        Args:
            Dthr (float): Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
                to a p-value of 1e-6.
            backend (str): 'cython' (default), 'jax', or 'auto' (prefer JAX if available).
            batch_size (int, optional): batch size used by the JAX backend to reduce peak memory usage.
            n_jobs (int, optional): number of threads for the Cython parallel backend.

        """
        return self._compute_kstar(
            Dthr=Dthr,
            backend=backend,
            batch_size=batch_size,
            n_jobs=n_jobs
        )

    # ----------------------------------------------------------------------------------------------

    def _compute_kstar_jax(self, Dthr=23.92812698, batch_size=None):
        if not _HAS_JAX:
            raise ModuleNotFoundError(
                "JAX is required for backend='jax'. Install `jax` and `jaxlib`."
            )

        if self.maxk <= 1:
            return np.ones(self.N, dtype=int)

        if self.maxk <= 4:
            return np.full(self.N, self.maxk - 1, dtype=int)

        if batch_size is None:
            batch_size = self.N
        batch_size = int(max(1, min(batch_size, self.N)))

        id_sel = float(self.intrinsic_dim)
        prefactor = np.exp(
            id_sel / 2.0 * np.log(np.pi) - gammaln((id_sel + 2.0) / 2.0)
        )

        dist_indices = jnp.asarray(self.dist_indices.astype(np.int64, copy=False))
        distances = jnp.asarray(self.distances.astype(np.float64, copy=False))
        j_values = jnp.arange(4, self.maxk, dtype=dist_indices.dtype)
        ksels = j_values - 1

        kstar = np.empty(self.N, dtype=np.int64)
        for start in range(0, self.N, batch_size):
            stop = min(start + batch_size, self.N)
            rows = jnp.arange(start, stop, dtype=dist_indices.dtype)
            row_grid = rows[:, None]

            vvi = prefactor * jnp.power(distances[row_grid, ksels[None, :]], id_sel)
            neigh_rows = dist_indices[row_grid, j_values[None, :]]
            vvj = prefactor * jnp.power(distances[neigh_rows, ksels[None, :]], id_sel)

            dL = -2.0 * ksels[None, :] * (
                jnp.log(vvi)
                + jnp.log(vvj)
                - 2.0 * jnp.log(vvi + vvj)
                + np.log(4.0)
            )
            reached = dL >= Dthr
            first_reached = jnp.argmax(reached, axis=1)
            has_reached = jnp.any(reached, axis=1)
            batch_kstar = jnp.where(has_reached, first_reached + 2, self.maxk - 1)
            kstar[start:stop] = np.asarray(batch_kstar, dtype=np.int64)

        return kstar

    # ----------------------------------------------------------------------------------------------

    def _compute_kstar(
        self, Dthr=23.92812698, backend="cython", batch_size=None, n_jobs=None
    ):
        """Compute an optimal choice of the neighbourhood size k for each point.

        Args:
            Dthr (float): likelihood-ratio threshold.
            backend (str): 'cython', 'jax', or 'auto'.
            batch_size (int or None): used only by backend='jax'.
            n_jobs (int or None): used by backend='cython' if an OpenMP-enabled Cython kernel is available.
        """
        if self.intrinsic_dim is None:
            warnings.warn(
                "Careful! The intrinsic dimension is not defined. "
                "Computing it unsupervisedly with 'compute_id_2NN()' method"
            )
            _ = self.compute_id_2NN()

        if self.distances is None or self.dist_indices is None:
            self.compute_distances()

        if backend not in {"cython", "jax", "auto"}:
            raise ValueError("backend must be one of {'cython', 'jax', 'auto'}")

        backend_resolved = backend
        if backend_resolved == "auto":
            backend_resolved = "jax" if _HAS_JAX else "cython"

        if self.verb:
            print(
                f"kstar estimation started, Dthr = {Dthr}, backend = '{backend_resolved}'"
            )

        sec = time.time()

        if backend_resolved == "jax":
            kstar = self._compute_kstar_jax(Dthr=Dthr, batch_size=batch_size)
        else:
            dist_indices = self.dist_indices.astype(np.int64, copy=False)
            distances = self.distances.astype(np.float64, copy=False)
            threads = self.n_jobs if n_jobs is None else n_jobs
            if (
                threads is not None
                and threads > 1
                and hasattr(cd, "_compute_kstar_parallel")
            ):
                kstar = cd._compute_kstar_parallel(
                    self.intrinsic_dim,
                    self.N,
                    self.maxk,
                    Dthr,
                    dist_indices,
                    distances,
                    int(threads),
                )
            else:
                kstar = cd._compute_kstar(
                    self.intrinsic_dim,
                    self.N,
                    self.maxk,
                    Dthr,
                    dist_indices,
                    distances,
                )

        self.set_kstar(kstar)

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing kstar".format(sec2 - sec))
