import numpy as np
from scipy.special import gammaln

from dadapy.cython_ import cython_maximum_likelihood_opt as cml


def return_density_kstarNN(distances, intrinsic_dim, kstar, interpolation=False):
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

    # Normalise density
    log_den -= np.log(N)

    log_den = log_den
    log_den_err = log_den_err

    return log_den, log_den_err, dc


def return_density_PAk(distances, intrinsic_dim, kstar, maxk, interpolation=False):
    N = distances.shape[0]

    dc = np.zeros(N, dtype=float)
    log_den = np.zeros(N, dtype=float)
    prefactor = np.exp(
        intrinsic_dim / 2.0 * np.log(np.pi) - gammaln((intrinsic_dim + 2.0) / 2.0)
    )
    log_den_min = 9.9e300

    if not interpolation:
        logkstars = np.log(kstar, dtype=float)
        log_den_err = 1.0 / np.sqrt(
            (4 * kstar + 2) / (kstar * (kstar - 1)), dtype=float
        )
    else:
        logkstars = np.log(kstar - 1, dtype=float)
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
            # to avoid easy overflow
            vi[j] = prefactor * (
                pow(distances[i, j + 1], intrinsic_dim)
                - pow(distances[i, j], intrinsic_dim)
            )

            if vi[j] < 1.0e-300:
                knn = 1
                break
        if knn == 0:
            log_den[i] = cml._nrmaxl(rr, kstar[i], vi, maxk)
        else:
            log_den[i] = rr
        if log_den[i] < log_den_min:
            log_den_min = log_den[i]

    # Normalise density
    log_den -= np.log(N)
    log_den = log_den
    log_den_err = log_den_err

    return log_den, log_den_err, dc
