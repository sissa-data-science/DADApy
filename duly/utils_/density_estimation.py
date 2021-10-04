import numpy as np
from scipy.special import gammaln
from duly.cython_ import cython_maximum_likelihood_opt as cml


def return_density_kstarNN(distances, intrinsic_dim, kstar):

    N = distances.shape[0]
    dc = np.zeros(N, dtype=float)
    log_den = np.zeros(N, dtype=float)
    log_den_err = np.zeros(N, dtype=float)
    prefactor = np.exp(
        intrinsic_dim / 2.0 * np.log(np.pi) - gammaln((intrinsic_dim + 2) / 2)
    )
    log_den_min = 9.9e300

    for i in range(N):
        dc[i] = distances[i, kstar[i]]
        log_den[i] = np.log(kstar[i]) - (
            np.log(prefactor) + intrinsic_dim * np.log(distances[i, kstar[i]])
        )

        log_den_err[i] = 1.0 / np.sqrt(kstar[i])
        if log_den[i] < log_den_min:
            log_den_min = log_den[i]

    # Normalise density
    log_den -= np.log(N)

    log_den = log_den
    log_den_err = log_den_err

    return log_den, log_den_err, dc


def return_density_PAk(distances, intrinsic_dim, kstar, maxk):

    N = distances.shape[0]

    dc = np.zeros(N, dtype=float)
    log_den = np.zeros(N, dtype=float)
    log_den_err = np.zeros(N, dtype=float)
    prefactor = np.exp(
        intrinsic_dim / 2.0 * np.log(np.pi) - gammaln((intrinsic_dim + 2.0) / 2.0)
    )
    log_den_min = 9.9e300

    for i in range(N):
        vi = np.zeros(maxk, dtype=float)
        dc[i] = distances[i, kstar[i]]
        rr = np.log(kstar[i]) - (
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

        log_den_err[i] = np.sqrt((4 * kstar[i] + 2) / (kstar[i] * (kstar[i] - 1)))

    # Normalise density
    log_den -= np.log(N)
    log_den = log_den
    log_den_err = log_den_err

    return log_den, log_den_err, dc