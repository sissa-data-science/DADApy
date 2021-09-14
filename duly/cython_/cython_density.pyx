import cython
import numpy as np
cimport numpy as np

from duly.cython_ import cython_maximum_likelihood_opt as cml

from scipy.special import gammaln
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport pi
from libc.math cimport sqrt
from libc.math cimport pow

DTYPE = np.int
floatTYPE = np.float
boolTYPE = np.bool

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

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_pak(floatTYPE_t id_selected,
                    DTYPE_t Nele,
                    DTYPE_t maxk,
                    np.ndarray[DTYPE_t, ndim = 1] kstar,
                    np.ndarray[floatTYPE_t, ndim = 2] distances
):

    cdef DTYPE_t i, j, knn
    cdef floatTYPE_t rr
    cdef floatTYPE_t Rho_min = 9.9e300
    cdef np.ndarray[floatTYPE_t, ndim = 1] vi = np.empty(maxk, dtype=float)
    cdef np.ndarray[floatTYPE_t, ndim = 1] dc = np.empty(Nele, dtype=float)
    cdef np.ndarray[floatTYPE_t, ndim = 1] Rho = np.empty(Nele, dtype=float)
    cdef np.ndarray[floatTYPE_t, ndim = 1] Rho_err = np.empty(Nele, dtype=float)

    cdef floatTYPE_t prefactor = exp( id_selected / 2.0 * log(pi) - gammaln((id_selected + 2.0) / 2.0) )

    for i in range(Nele):
        dc[i] = distances[i, kstar[i]]
        rr = log(kstar[i]) - ( log(prefactor) + id_selected * log(distances[i, kstar[i]] )            )
        knn = 0
        for j in range(kstar[i]):
            # to avoid easy overflow
            vi[j] = prefactor * (
                pow(distances[i, j + 1], id_selected)
                - pow(distances[i, j], id_selected)
            )
            if vi[j] < 1.0e-300:
                knn = 1
                break
        if knn == 0:
            #if method == "NR":
            Rho[i] = cml._nrmaxl(rr, kstar[i], vi, maxk)
            #elif method == "NM":
            #    from duly.utils_.mlmax import MLmax

            #    log_den[i] = MLmax(rr, self.kstar[i], vi)
            #else:
            #    raise ValueError("Please choose a valid method")
            # log_den[i] = NR.nrmaxl(rr, kstar[i], vi, self.maxk) # OLD FORTRAN
        else:
            Rho[i] = rr
        if Rho[i] < Rho_min:
            Rho_min = Rho[i]

        Rho_err[i] = sqrt(
            (4 * kstar[i] + 2) / (kstar[i] * (kstar[i] - 1))
        )
        # Normalise density
        Rho[i] -= log(Nele)

    return Rho, Rho_err, dc