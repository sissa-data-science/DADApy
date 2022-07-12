import cython
import numpy as np

cimport numpy as np
from cython.parallel cimport parallel, prange
from libc.math cimport (  # absolute values for floats, needed when using PBC
    fabs,
    nearbyint,
)
from libc.stdlib cimport abs  # absolute value for integers

#DTYPE = np.int
#floatTYPE = np.float
#boolTYPE = np.bool

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

#------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def _return_hamming_condensed(np.ndarray[DTYPE_t, ndim = 2] points,
                            int d_max):

    cdef int N = len(points)
    cdef int L = len(points[0])
    cdef int i, j, k, appo
    cdef np.ndarray[DTYPE_t, ndim = 2] distances = np.zeros((N, d_max + 1), dtype=np.int)

    for i in range(N):
        distances[i,0] += 1
        for j in range(i+1,N):
            appo = 0
            for k in range(L):
                if points[i,k] != points[j,k]:
                    appo += 1
            if appo <= d_max:
                distances[i, appo] += 1
                distances[j, appo] += 1
        for k in range(1,d_max+1):
            distances[i,k] += distances[i,k-1]

    return distances


#------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def _return_hamming_condensed_parallel(np.ndarray[DTYPE_t, ndim = 2] points,
                            int d_max,
                            int n_jobs):

    cdef long N = len(points)
    cdef long L = len(points[0])
    cdef int i, j, k, ind
    cdef np.ndarray[DTYPE_t, ndim = 2] distances = np.zeros((N, d_max + 1), dtype=np.int)
    cdef long [:,:] dv = distances
    cdef long [:,:] pv = points

    with nogil, parallel(num_threads=n_jobs):
        for i in prange(N, schedule='static'):
            for j in range(N):
                ind = 0
                for k in range(L):
                    if pv[i,k] != pv[j,k]:
                        ind = ind + 1
                if ind <= d_max:
                    dv[i, ind] = dv[i, ind] + 1
            for k in range(1,d_max+1):
                dv[i,k] += dv[i,k-1]

    return distances
#------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def _return_manhattan_condensed(np.ndarray[DTYPE_t, ndim = 2] points,
                            int d_max,
                            np.ndarray[floatTYPE_t, ndim = 1] period):

    cdef int N = len(points)
    cdef int L = len(points[0])
    cdef int i, j, k, ind
    cdef double d, appo
    cdef np.ndarray[DTYPE_t, ndim = 2] distances = np.zeros((N, d_max + 1), dtype=int)

    if period is None:
        for i in range(N):
            distances[i,0] += 1
            for j in range(i+1,N):
                appo = 0
                for k in range(L):
                    appo += abs(points[i,k]-points[j,k])
                if appo <= d_max:
                    ind = int(appo)
                    distances[i, ind] += 1
                    distances[j, ind] += 1
            for k in range(1,d_max+1):
                distances[i,k] += distances[i,k-1]

    else:
        for i in range(N):
            distances[i, 0] += 1
            for j in range(i+1, N):
                appo = 0
                for k in range(L):
                    d = points[i, k] - points[j, k]
                    appo += fabs(d - nearbyint(d/period[k])*period[k])
                if appo <= d_max:
                    ind = int(appo)
                    distances[i, ind] += 1
                    distances[j, ind] += 1
            for k in range(1, d_max + 1):
                distances[i, k] += distances[i, k - 1]

    return distances


#------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def _return_manhattan_condensed_parallel(np.ndarray[DTYPE_t, ndim = 2] points,
                            int d_max,
                            np.ndarray[floatTYPE_t, ndim = 1] period,
                            int n_jobs):

    cdef int N = points.shape[0]
    cdef int L = points.shape[1]

    cdef int i,j,k,l
    cdef int ind
    cdef double appo, d

    cdef np.ndarray[DTYPE_t, ndim = 2] distances = np.zeros((N, d_max + 1), dtype=np.int)
    cdef long [:,:] dv = distances
    cdef long [:,:] pv = points
    cdef double [:] period_v = period

    if period is None:
        with nogil, parallel(num_threads=n_jobs):
            for i in prange(N, schedule='static'):
                for j in range(N):
                    ind = 0
                    for k in range(L):
                        ind = ind + abs(pv[i, k] - pv[j, k])
                    if ind <= d_max:
                        dv[i, ind] = dv[i, ind] + 1
                for l in range(1,d_max+1):
                    dv[i,l] += dv[i,l-1]
    else:
        with nogil, parallel(num_threads=n_jobs):
            for i in prange(N, schedule='static'):
                for j in range(N):
                    appo = 0
                    for k in range(L):
                        d = pv[i, k] - pv[j, k]
                        appo = appo + fabs(d - nearbyint(d/period_v[k])*period_v[k])
                    if ind <= d_max:
                        ind = int(appo)
                        dv[i, ind] = dv[i, ind] + 1
                for l in range(1,d_max+1):
                    dv[i,l] += dv[i,l-1]

    return distances

"""
    # tentative of doing N**2/2 operation by using the indices from upper triangular matrix
    # still having some memory overlaps

    idxx = np.triu_indices(N)
    cdef np.ndarray[DTYPE_t, ndim = 1] I = idxx[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] J = idxx[1]
    cdef long [:] iv = I
    cdef long [:] jv = J
    cdef int NN = I.shape[0]


    with nogil, parallel(num_threads=8):
        for i in prange(NN, schedule='static'):
            ind = 0
            for k in range(L):
                ind = ind + abs(pv[iv[i],k]-pv[jv[i],k])
            if ind <= d_max:
                dv[iv[i], ind] = dv[iv[i], ind] + 1
                dv[jv[i], ind] = dv[jv[i], ind] + 1

    for i in range(N):
        dv[i,0] = dv[i,0] - 1
        for l in range(1,d_max+1):
            dv[i,l] = dv[i,l] + dv[i,l-1]

    return distances
"""