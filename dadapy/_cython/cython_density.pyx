# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cython
import numpy as np

cimport numpy as np
from cython.parallel cimport prange

from scipy.special import gammaln
from scipy.stats import chi2

from libc.math cimport exp, log, pi, pow

DTYPE = np.int64
floatTYPE = np.float64


ctypedef np.int64_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_kstar(floatTYPE_t id_sel,
                    DTYPE_t Nele,
                    DTYPE_t maxk,
                    floatTYPE_t alpha,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                    np.ndarray[floatTYPE_t, ndim = 2] distances,
                    bint bonferroni_deloc,
                    bint bonferroni_loc,
):
    """Likelihhod ratio test to find the k*, ie the farthest neighbour for which the density is
    considered approximately constant. We include also bonferroni correction for multiple testing,
    in the delocalised version (alpha scaled by number of points) and the local one. In principle, for
    the local one, the threshold is updated at each new test and in principle one should check that
    all dL passes test with the new threshold. However, since the threshold is always increasing (as alpha is reduced),
    practically one has to check only for the last dL (againt the proper threshold).
    """

    cdef floatTYPE_t dL, vvi, vvj, thr
    cdef DTYPE_t i, j, ksel, h
    cdef np.ndarray[DTYPE_t, ndim = 1] kstar = np.empty(Nele, dtype=int)
    cdef floatTYPE_t prefactor = exp( id_sel / 2.0 * log(pi) - gammaln((id_sel + 2.0) / 2.0) )
    cdef floatTYPE_t alpha_eff = alpha / Nele if bonferroni_deloc else alpha
    cdef floatTYPE_t threshold = chi2.isf(alpha_eff, 1)
    cdef np.ndarray[floatTYPE_t, ndim = 1] local_thresholds = np.asarray(
        chi2.isf(alpha_eff / np.arange(1, maxk + 1), 1), dtype=float
    )

    if maxk <= 1:
        kstar[:] = 1
        return kstar
    if maxk <= 4:
        kstar[:] = maxk - 1
        return kstar

    for i in range(Nele):
        j = 4
        dL = 0.0
        h = 0
        while j < maxk:
            ksel = j - 1
            vvi = prefactor * pow(distances[i, ksel], id_sel)
            vvj = prefactor * pow(distances[dist_indices[i, j], ksel], id_sel)
            dL = -2.0 * ksel * ( log(vvi) + log(vvj) - 2.0 * log(vvi + vvj) + log(4) )
            thr = local_thresholds[h] if bonferroni_loc else threshold
            if dL > thr:
                break
            else:
                j = j + 1
                h = h + 1
        kstar[i] = j - 1  # fall back to previous iteration where the test had passed

    return kstar


@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_kstar_parallel(floatTYPE_t id_sel,
                    DTYPE_t Nele,
                    DTYPE_t maxk,
                    floatTYPE_t alpha,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                    np.ndarray[floatTYPE_t, ndim = 2] distances,
                    bint bonferroni_deloc,
                    bint bonferroni_loc,
                    DTYPE_t n_jobs):
    """Parallel version of the likelihood-ratio kstar test."""

    cdef floatTYPE_t dL, vvi, vvj, thr
    cdef DTYPE_t i, j, ksel, h
    cdef np.ndarray[DTYPE_t, ndim = 1] kstar = np.empty(Nele, dtype=int)
    cdef floatTYPE_t prefactor = exp( id_sel / 2.0 * log(pi) - gammaln((id_sel + 2.0) / 2.0) )
    cdef floatTYPE_t alpha_eff = alpha / Nele if bonferroni_deloc else alpha
    cdef floatTYPE_t threshold = chi2.isf(alpha_eff, 1)
    cdef np.ndarray[floatTYPE_t, ndim = 1] local_thresholds = np.asarray(
        chi2.isf(alpha_eff / np.arange(1, maxk + 1), 1), dtype=float
    )

    cdef DTYPE_t[:, ::1] dist_indices_v = dist_indices
    cdef floatTYPE_t[:, ::1] distances_v = distances
    cdef DTYPE_t[::1] kstar_v = kstar
    cdef floatTYPE_t[::1] local_thresholds_v = local_thresholds

    if maxk <= 1:
        kstar[:] = 1
        return kstar
    if maxk <= 4:
        kstar[:] = maxk - 1
        return kstar

    with nogil:
        for i in prange(Nele, schedule='static', num_threads=n_jobs):
            j = 4
            dL = 0.0
            h = 0
            while j < maxk:
                ksel = j - 1
                vvi = prefactor * pow(distances_v[i, ksel], id_sel)
                vvj = prefactor * pow(distances_v[dist_indices_v[i, j], ksel], id_sel)
                dL = -2.0 * ksel * ( log(vvi) + log(vvj) - 2.0 * log(vvi + vvj) + log(4) )
                thr = local_thresholds_v[h] if bonferroni_loc else threshold
                if dL > thr:
                    break
                j = j + 1
                h = h + 1
            kstar_v[i] = j - 1

    return kstar


@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_kstar_interp(floatTYPE_t id_sel,
                          DTYPE_t Nele,
                          DTYPE_t maxk,
                          floatTYPE_t alpha,
                          np.ndarray[DTYPE_t, ndim = 2] cross_dist_indices,
                          np.ndarray[floatTYPE_t, ndim = 2] cross_distances,
                          np.ndarray[floatTYPE_t, ndim = 2] data_distances,
                          bint bonferroni_deloc,
                          bint bonferroni_loc,
                          ):
    """Find kstar for query points with optional Bonferroni corrections."""

    cdef floatTYPE_t dL, vvi, vvj, thr
    cdef DTYPE_t i, j, ksel, h
    cdef np.ndarray[DTYPE_t, ndim = 1] kstar = np.empty(Nele, dtype=int)
    cdef floatTYPE_t prefactor = exp( id_sel / 2.0 * log(pi) - gammaln((id_sel + 2.0) / 2.0) )
    cdef floatTYPE_t alpha_eff = alpha / Nele if bonferroni_deloc else alpha
    cdef floatTYPE_t threshold = chi2.isf(alpha_eff, 1)
    cdef np.ndarray[floatTYPE_t, ndim = 1] local_thresholds = np.asarray(
        chi2.isf(alpha_eff / np.arange(1, maxk + 1), 1), dtype=float
    )

    if maxk <= 1:
        kstar[:] = 1
        return kstar
    if maxk <= 4:
        kstar[:] = maxk - 1
        return kstar

    for i in range(Nele):
        j = 4
        dL = 0.0
        h = 0
        while j < maxk:
            ksel = j - 1
            vvi = prefactor * pow(cross_distances[i, ksel], id_sel)
            vvj = prefactor * pow(data_distances[cross_dist_indices[i, j], ksel], id_sel)
            dL = -2.0 * ksel * ( log(vvi) + log(vvj) - 2.0 * log(vvi + vvj) + log(4) )
            thr = local_thresholds[h] if bonferroni_loc else threshold
            if dL > thr:
                break
            else:
                j = j + 1
                h = h + 1
        kstar[i] = j - 1

    return kstar
