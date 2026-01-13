# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np

from scipy.stats import chi2

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
                    floatTYPE_t alpha,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                    np.ndarray[floatTYPE_t, ndim = 2] distances,
                    bint bonferroni_deloc,
                    bint bonferroni_loc,
):

    cdef floatTYPE_t dL, vvi, vvj
    cdef DTYPE_t i, j, ksel, h
    cdef np.ndarray[DTYPE_t, ndim = 1] kstar = np.empty(Nele, dtype=int)
    cdef floatTYPE_t prefactor = exp( id_sel / 2.0 * log(pi) - gammaln((id_sel + 2.0) / 2.0) )
    cdef floatTYPE_t alpha_eff = alpha / Nele if bonferroni_deloc else alpha
    cdef floatTYPE_t Dthr = chi2.isf(alpha_eff,1)
    cdef np.ndarray[floatTYPE_t, ndim = 1] Dthr_loc_arr = np.array([chi2.isf(alpha_eff / (h+1), 1) for h in range(maxk)], dtype=float)
    cdef np.ndarray[floatTYPE_t, ndim = 1] dL_arr = np.empty(maxk, dtype=float)
    cdef bint all_pass

    for i in range(Nele):
        j = 4
        dL = 0.0
        h = 0
        if bonferroni_loc:
            while j < maxk:
                ksel = j - 1
                vvi = prefactor * pow(distances[i, ksel], id_sel)
                vvj = prefactor * pow(distances[dist_indices[i, j], ksel], id_sel)
                dL = -2.0 * ksel * ( log(vvi) + log(vvj) - 2.0 * log(vvi + vvj) + log(4) )
                dL_arr[h] = dL
                # Check all previous dL against current Dthr_loc
                all_pass = True
                for k in range(h+1):
                    if dL_arr[k] > Dthr_loc_arr[h]:  # stop if any dL exceeds threshold
                        all_pass = False
                        break
                if not all_pass:
                    break
                h += 1
                j += 1
            if j == 4:
                kstar[i] = 3
            else:
                kstar[i] = j - 2
        else:
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
