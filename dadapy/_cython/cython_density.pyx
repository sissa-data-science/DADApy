# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cython
import numpy as np

cimport numpy as np
from cython.parallel cimport prange

from scipy.special import gammaln

from libc.math cimport exp, log, pi, pow

DTYPE = np.int64
floatTYPE = np.float64


ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_kstar(floatTYPE_t id_sel,
                    DTYPE_t Nele,
                    DTYPE_t maxk,
                    floatTYPE_t Dthr,#=23.92812698,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                    np.ndarray[floatTYPE_t, ndim = 2] distances
):


    cdef floatTYPE_t dL, vvi, vvj
    cdef DTYPE_t i, j, ksel
    cdef np.ndarray[DTYPE_t, ndim = 1] kstar = np.empty(Nele, dtype=int)
    cdef floatTYPE_t prefactor = exp( id_sel / 2.0 * log(pi) - gammaln((id_sel + 2.0) / 2.0) )

    for i in range(Nele):
        j = 4
        dL = 0.0
        while j < maxk and dL < Dthr:
            ksel = j - 1
            vvi = prefactor * pow(distances[i, ksel], id_sel)
            vvj = prefactor * pow(distances[dist_indices[i, j], ksel], id_sel)
            dL = -2.0 * ksel * ( log(vvi) + log(vvj) - 2.0 * log(vvi + vvj) + log(4) )
            j = j + 1
        if j == maxk:
            kstar[i] = j - 1
        else:
            kstar[i] = j - 2

    return kstar


@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_kstar_parallel(floatTYPE_t id_sel,
                    DTYPE_t Nele,
                    DTYPE_t maxk,
                    floatTYPE_t Dthr,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                    np.ndarray[floatTYPE_t, ndim = 2] distances,
                    DTYPE_t n_jobs):


    cdef floatTYPE_t dL, vvi, vvj
    cdef DTYPE_t i, j, ksel
    cdef np.ndarray[DTYPE_t, ndim = 1] kstar = np.empty(Nele, dtype=int)
    cdef floatTYPE_t prefactor = exp( id_sel / 2.0 * log(pi) - gammaln((id_sel + 2.0) / 2.0) )

    cdef DTYPE_t[:, ::1] dist_indices_v = dist_indices
    cdef floatTYPE_t[:, ::1] distances_v = distances
    cdef DTYPE_t[::1] kstar_v = kstar

    with nogil:
        for i in prange(Nele, schedule='static', num_threads=n_jobs):
            j = 4
            dL = 0.0
            while j < maxk and dL < Dthr:
                ksel = j - 1
                vvi = prefactor * pow(distances_v[i, ksel], id_sel)
                vvj = prefactor * pow(distances_v[dist_indices_v[i, j], ksel], id_sel)
                dL = -2.0 * ksel * ( log(vvi) + log(vvj) - 2.0 * log(vvi + vvj) + log(4) )
                j = j + 1
            if j == maxk:
                kstar_v[i] = j - 1
            else:
                kstar_v[i] = j - 2

    return kstar


@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_kstar_interp(floatTYPE_t id_sel,
                          DTYPE_t Nele,
                          DTYPE_t maxk,
                          floatTYPE_t Dthr,  #=23.92812698,
                          np.ndarray[DTYPE_t, ndim = 2] cross_dist_indices,
                          np.ndarray[floatTYPE_t, ndim = 2] cross_distances,
                          np.ndarray[floatTYPE_t, ndim = 2] data_distances
                          ):


    cdef floatTYPE_t dL, vvi, vvj
    cdef DTYPE_t i, j, ksel
    cdef np.ndarray[DTYPE_t, ndim = 1] kstar = np.empty(Nele, dtype=int)
    cdef floatTYPE_t prefactor = exp( id_sel / 2.0 * log(pi) - gammaln((id_sel + 2.0) / 2.0) )

    for i in range(Nele):
        j = 4
        dL = 0.0
        while j < maxk and dL < Dthr:
            ksel = j - 1
            vvi = prefactor * pow(cross_distances[i, ksel], id_sel)
            vvj = prefactor * pow(data_distances[cross_dist_indices[i, j], ksel], id_sel)
            dL = -2.0 * ksel * ( log(vvi) + log(vvj) - 2.0 * log(vvi + vvj) + log(4) )
            j = j + 1
        if j == maxk:
            kstar[i] = j - 1
        else:
            kstar[i] = j - 2

    return kstar
