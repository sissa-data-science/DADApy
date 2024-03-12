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
import copy
import warnings

import numpy as np
from scipy.special import gammaln

from dadapy._cython import cython_maximum_likelihood_opt as cml
from dadapy._cython import cython_maximum_likelihood_opt_full as cml_full


def return_not_normalised_density_kstarNN(
    distances,
    intrinsic_dim,
    kstar,
    interpolation=False,
    bias=False,
):
    N = distances.shape[0]
    dc = np.zeros(N, dtype=float)
    prefactor = np.exp(
        intrinsic_dim / 2.0 * np.log(np.pi) - gammaln((intrinsic_dim + 2) / 2)
    )
    log_den_min = 9.9e300

    if not interpolation:
        log_den = np.log(kstar, dtype=float)
        log_den_err = 1.0 / np.sqrt(kstar, dtype=float)

    else:
        log_den = np.log(kstar - 1, dtype=float)
        log_den_err = 1.0 / np.sqrt(kstar - 1, dtype=float)
    if bias:
        warnings.warn(
            "bias contribution to the density error is an experimental feature \
            and might change in the future"
        )
        log_den_err = (log_den_err**2 + (kstar / N) ** (2 / intrinsic_dim)) ** 0.5

    for i in range(N):
        dc[i] = distances[i, kstar[i]]
        log_den[i] = log_den[i] - (
            np.log(prefactor) + intrinsic_dim * np.log(distances[i, kstar[i]])
        )

        if log_den[i] < log_den_min:
            log_den_min = log_den[i]

    return log_den, log_den_err, dc


def return_not_normalised_density_PAk(
    distances, intrinsic_dim, kstar, interpolation=False, bias=False
):
    N = distances.shape[0]

    log_den = np.zeros(N, dtype=float)
    prefactor = np.exp(
        intrinsic_dim / 2.0 * np.log(np.pi) - gammaln((intrinsic_dim + 2.0) / 2.0)
    )
    log_den_min = 9.9e300

    if not interpolation:
        logkstars = np.log(kstar, dtype=float)
        log_den_err = np.sqrt((4 * kstar + 2) / (kstar * (kstar - 1)), dtype=float)
    else:
        logkstars = 1.0 / np.log(kstar - 1, dtype=float)
        log_den_err = np.sqrt(
            (4 * (kstar - 1) + 2) / ((kstar - 1) * ((kstar - 1) - 1)), dtype=float
        )

    if bias:
        warnings.warn(
            "bias contribution to the density error is an experimental \
            feature and might change in the future"
        )
        log_den_err = (log_den_err**2 + (kstar / N) ** (2 / intrinsic_dim)) ** 0.5

    dc = distances[np.arange(N), kstar]

    if intrinsic_dim * np.log(np.max(dc)) > np.log(np.max(np.finfo(np.float64).max)):
        warnings.warn(
            "Some volumes (r^intrisic_dim) may cause overflow: those values will \
            be silently set to e^300."
        )
        noverflows = np.sum(
            intrinsic_dim * np.log(dc) + np.log(prefactor)
            > np.log(np.max(np.finfo(np.float64).max))
        )
        print(
            f"intrinsic_dim = {intrinsic_dim}, rmax = {np.max(dc)}, rmin = {np.min(dc)}, float type = {dc[0].dtype}, fraction of potential overflows = {noverflows}/{N}"
        )

    max_ratio = 0.0
    for i in range(N):
        vi = np.zeros(max(kstar), dtype=np.float64)

        rr = logkstars[i] - (
            np.log(prefactor) + intrinsic_dim * np.log(distances[i, kstar[i]])
        )

        knn = 0

        for j in range(kstar[i]):
            r = distances[i, j]
            r1 = distances[i, j + 1]
            ratio = r / r1
            if ratio > max_ratio:
                max_ratio = ratio
            if np.abs(ratio - 1.0) < np.finfo(r.dtype).resolution:
                warnings.warn(
                    "Found nearest neighbours at identical distance, adding a small \
                    amount of noise to one distance."
                )
                ratio -= 10 * np.finfo(r.dtype).resolution

            exponent = intrinsic_dim * np.log(r1) + np.log(1 - ratio**intrinsic_dim)

            if exponent > 300:
                vi[j] = prefactor * np.exp(300.0)
            else:
                vi[j] = prefactor * np.exp(exponent)

            if vi[j] < 1.0e-300:
                knn = 1
                break

        if knn == 0:
            log_den[i] = cml._nrmaxl(rr, kstar[i], vi)
        else:
            log_den[i] = rr
        if log_den[i] < log_den_min:
            log_den_min = log_den[i]

    return log_den, log_den_err, dc


def return_not_normalised_density_PAk_optimized(
    distances, intrinsic_dim, kstar, interpolation=False, bias=False
):
    N = distances.shape[0]
    if not interpolation:
        logkstars = np.log(kstar, dtype=float)
        log_den_err = np.sqrt((4 * kstar + 2) / (kstar * (kstar - 1)), dtype=float)
    else:
        logkstars = 1.0 / np.log(kstar - 1, dtype=float)
        log_den_err = np.sqrt(
            (4 * (kstar - 1) + 2) / ((kstar - 1) * ((kstar - 1) - 1)), dtype=float
        )
    if bias:
        log_den_err = (log_den_err**2 + (kstar / N) ** (2 / intrinsic_dim)) ** 0.5

    dc = distances[np.arange(N), kstar]

    prefactor = np.exp(
        intrinsic_dim / 2.0 * np.log(np.pi) - gammaln((intrinsic_dim + 2.0) / 2.0)
    )

    indices_radii = np.arange(max(kstar) + 1)

    r = distances[:, indices_radii[:-1]]
    r1 = distances[:, indices_radii[1:]]
    ratio = r / r1

    mask = np.abs(ratio - 1.0) < np.finfo(r.dtype).resolution

    if np.any(mask):
        ratio[mask] -= 10 * np.finfo(r.dtype).resolution
        nidentical = 0
        for i in range(N):
            nidentical += np.sum(mask[i, : kstar[i] + 1])

        if nidentical > 0:
            warnings.warn(
                f"Found {nidentical} nearest neighbours at identical distance, adding a small amount of noise"
            )

    exponent = intrinsic_dim * np.log(r1) + np.log(1 - ratio**intrinsic_dim)
    overflow = exponent > 300.0

    if np.any(overflow):
        warnings.warn(
            f"ID too high. Found {np.sum(overflow)} shell volumes > e^300: settting volumes to e^300"
        )

        exponent[overflow] = 300.0  # + np.random.normal(size=(np.sum(overflow)))

    volumes = prefactor * np.exp(exponent)

    # caluculation of the NEGATIVE free energy that maximizes the likelihood
    starting_roots = logkstars - (
        np.log(np.repeat(prefactor, N)) + intrinsic_dim * np.log(dc)
    )

    log_den, is_singular = cml_full._nrmaxl(
        copy.deepcopy(starting_roots), kstar, volumes
    )

    if is_singular:
        warnings.warn(
            "Hessian matrix in NR max likelihood maximization is sigular: using fixed point step"
        )

    return log_den, log_den_err, dc
