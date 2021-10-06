
# cython implementation of the distance computation in periodic boundary conditions code taken from:
# https://www.guangshi.io/posts/python-optimization-using-different-methods-part-3#serial-cython-implementation

import cython
import numpy as np

from libc.math cimport sqrt
from libc.math cimport nearbyint

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # Do not check division, may leads to 20% performance speedup
def pdist_serial(double [:,:] positions not None, double l):
    cdef Py_ssize_t n = positions.shape[0]
    cdef Py_ssize_t ndim = positions.shape[1]

    pdistances = np.zeros(n * (n-1) // 2, dtype = np.float64)
    cdef double [:] pdistances_view = pdistances

    cdef double d, dd
    cdef Py_ssize_t i, j, k

    for i in range(n-1):
        for j in range(i+1, n):
            dd = 0.0
            for k in range(ndim):
                d = positions[i,k] - positions[j,k]
                d = d - nearbyint(d / l) * l
                dd += d * d

            pdistances_view[j - 1 + (2 * n - 3 - i) * i // 2] = sqrt(dd)

    return pdistances


# ### parallel implementation ###
#
# from cython.parallel cimport prange, parallel
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True) # Do not check division, may leads to 20% performance speedup
# def pdist_parallel(double [:,:] positions not None, double l):
#     cdef Py_ssize_t n = positions.shape[0]
#     cdef Py_ssize_t ndim = positions.shape[1]
#
#     pdistances = np.zeros(n * (n-1) // 2, dtype = np.float64)
#     cdef double [:] pdistances_view = pdistances
#
#     cdef double d, dd
#     cdef Py_ssize_t i, j, k
#     with nogil, parallel():
#         for i in prange(n - 1, schedule='dynamic'):
#             for j in range(i+1, n):
#                 dd = 0.0
#                 for k in range(ndim):
#                     d = positions[i,k] - positions[j,k]
#                     d = d - nearbyint(d / l) * l
#                     dd = dd + d * d
#                 pdistances_view[j - 1 + (2 * n - 3 - i) * i // 2] = sqrt(dd)
#
#     return pdistances
