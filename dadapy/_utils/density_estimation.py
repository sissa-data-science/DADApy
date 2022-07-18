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

import time

import numpy as np
from scipy.special import gammaln

from dadapy._cython import cython_maximum_likelihood_opt as cml
from dadapy._cython import cython_maximum_likelihood_opt_full as cml_full


def return_not_normalised_density_kstarNN(
    distances,
    intrinsic_dim,
    kstar,
    interpolation=False,
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

    for i in range(N):
        dc[i] = distances[i, kstar[i]]
        log_den[i] = log_den[i] - (
            np.log(prefactor) + intrinsic_dim * np.log(distances[i, kstar[i]])
        )

        if log_den[i] < log_den_min:
            log_den_min = log_den[i]

    return log_den, log_den_err, dc


def return_not_normalised_density_PAk(
    distances, intrinsic_dim, kstar, maxk, interpolation=False
):
    N = distances.shape[0]

    dc = np.zeros(N, dtype=float)
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

    for i in range(N):
        vi = np.zeros(maxk, dtype=float)
        dc[i] = distances[i, kstar[i]]
        rr = logkstars[i] - (
            np.log(prefactor) + intrinsic_dim * np.log(distances[i, kstar[i]])
        )
        knn = 0
        for j in range(kstar[i]):

            vi[j] = prefactor * (
                pow(distances[i, j + 1], intrinsic_dim)
                - pow(distances[i, j], intrinsic_dim)
            )

            # to avoid easy overflow
            #   maybe try to add a warning to the previous implementation:
            #   in well behaved cases (e.g. IDs order of tens or lower) previous implementation
            #   should not overflow

            # r = distances[i, j]
            # r1 = distances[i, j + 1]
            # exponent = intrinsic_dim * np.log(r1) + np.log(
            #     1 - (r / r1) ** intrinsic_dim
            # )
            # vi[j] = prefactor * np.exp(exponent)

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
    distances, intrinsic_dim, kstar, maxk, interpolation=False
):
    if not interpolation:
        logkstars = np.log(kstar, dtype=float)
        log_den_err = np.sqrt((4 * kstar + 2) / (kstar * (kstar - 1)), dtype=float)
    else:
        logkstars = 1.0 / np.log(kstar - 1, dtype=float)
        log_den_err = np.sqrt(
            (4 * (kstar - 1) + 2) / ((kstar - 1) * ((kstar - 1) - 1)), dtype=float
        )

    N = distances.shape[0]
    dc = distances[np.arange(N), kstar]

    prefactor = np.exp(
        intrinsic_dim / 2.0 * np.log(np.pi) - gammaln((intrinsic_dim + 2.0) / 2.0)
    )
    indices_radii = np.arange(max(kstar) + 1)
    volumes = prefactor * (
        distances[:, indices_radii[1:]] ** intrinsic_dim
        - distances[:, indices_radii[:-1]] ** intrinsic_dim
    )

    # caluculation of the NEGATIVE free energy that maximizes the likelihood
    starting_roots = logkstars - (
        np.log(np.repeat(prefactor, N)) + intrinsic_dim * np.log(dc)
    )
    log_den = cml_full._nrmaxl(starting_roots, kstar, volumes)

    return log_den, log_den_err, dc


# alternative solution: much slower
# from scipy.optimize import minimize
# log_den = np.zeros(N)
# for i in range(N):
#     if np.any(volumes[i]<1.e-300):
#         log_den[i] = starting_roots[i]
#     else:
#         #log_den[i] = cml._nrmaxl(starting_roots[i], kstar[i], volumes[i, :kstar[i]])
#             start_maxl = time.time()
#             result = minimize(    x0 = np.array([starting_roots[i], 0.]),
#                                     fun = neg_lik,
#                                     jac=jac_neg_lik,
#                                     args=(kstar[i], volumes[i, :kstar[i]]),
#                                     method = 'BFGS',
#                                     tol = 1e-5,
#                                     options = {'maxiter': 10000}
#                                 )
#             log_den[i] = result.x[0]
#             end_maxl = time.time()
#             time_maxl += end_maxl-start_maxl


# def neg_lik(x, kstar, volumes):
#     #the negative likelihood is minimized as a function of -F variable (to avoid numerical overflows)
#     F, a = -x[0], x[1]
#     N = volumes.shape[0]
#     l = np.arange(1, N+1)
#     F_tmp = np.repeat(F, N)
#     lik = -F*kstar + 0.5*a*kstar*(kstar+1) - np.sum(volumes*np.exp(a*l -F_tmp))
#     return -lik
#
#
# def jac_neg_lik(x, kstar, volumes):
#
#     F, a = -x[0], x[1]
#     l = np.arange(1, volumes.shape[0]+1)
#
#     dF = kstar - np.exp(-F)*np.sum(volumes*np.exp(a*l))
#
#     da = 0.5*kstar*(kstar+1)-np.exp(-F)*np.sum(volumes*l*np.exp(a*l))
#
#     d_neg_lik = np.array([-dF, -da])
#
#     return d_neg_lik
