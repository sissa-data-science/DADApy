# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cython
import numpy as np

cimport numpy as np
from libc.math cimport exp

DTYPE = np.int64
floatTYPE = np.float64

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_data_overlap(DTYPE_t Nele,
                    DTYPE_t k,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices1,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices2,
):

    cdef np.ndarray[floatTYPE_t, ndim=1] overlaps = -np.ones(Nele)
    cdef floatTYPE_t count
    cdef DTYPE_t i, j, l
    cdef DTYPE_t[:, ::1] indices1 = dist_indices1
    cdef DTYPE_t[:, ::1] indices2 = dist_indices2

    for i in range(Nele):
        count = 0
        for j in range(1, k+1):
            elem = indices1[i, j]
            for l in range(1, k + 1):
                if elem == indices2[i, l]:
                    count += 1
                    break
        overlaps[i] = count / k

    return overlaps
