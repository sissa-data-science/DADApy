import cython
import numpy as np

cimport numpy as np
from cython.parallel cimport parallel, prange
from libc.math cimport nearbyint, fabs    # absolute values for floats, needed when using PBC
from libc.stdlib cimport abs              # absolute value for integers

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
    cdef np.ndarray[DTYPE_t, ndim = 2] distances = np.zeros((N, d_max + 1), dtype=int)

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
                            int d_max):

    cdef Py_ssize_t N = len(points)
    cdef Py_ssize_t L = len(points[0])
    cdef Py_ssize_t i, j, k, appo
    #cdef np.ndarray[DTYPE_t, ndim = 2] distances = np.zeros((N, d_max + 1), dtype=int)
    cdef distances = np.zeros((N, d_max + 1), dtype=int)
    cdef Py_ssize_t [:,:] distances_view = distances

    with nogil, parallel(num_threads=16):
        for i in prange(N, schedule='dynamic'):
            distances_view[i,0] += 1
            for j in range(i+1,N):
                appo = 0
                for k in range(L):
                    if points[i,k] != points[j,k]:
                        appo = appo + 1
                if appo <= d_max:
                    distances_view[i, appo] = distances_view[i, appo] + 1
                    distances_view[j, appo] = distances_view[j, appo] + 1
            for k in range(1,L):
                distances_view[i,k] += distances_view[i,k-1]

    return distances
#------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def _return_manhattan_condensed(np.ndarray[DTYPE_t, ndim = 2] points,
                            int d_max,
                            np.ndarray[DTYPE_t, ndim = 1] period):

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
                            np.ndarray[DTYPE_t, ndim = 1] period):

    cdef Py_ssize_t N = points.shape[0]
    cdef Py_ssize_t L = points.shape[1]

    cdef Py_ssize_t i, j, k, l
    cdef int appo, ind

    distances = np.zeros((N, d_max + 1), dtype=int)
    cdef long [:,:] distances_view = distances

#    cdef np.ndarray[DTYPE_t, ndim = 1] appo = np.zeros(N, dtype=int)
#    cdef long [:] appov = appo

#    cdef np.ndarray[DTYPE_t, ndim = 1] ind = np.zeros(N, dtype=int)
#    cdef long [:] indv = ind

#    cdef np.ndarray[DTYPE_t, ndim = 1] j = np.zeros(N, dtype=int)
#    cdef Py_ssize_t [:] jv = j



    if period is None:
        with nogil, parallel(num_threads=8):
            for i in prange(N, schedule='dynamic'):
                distances_view[i,0] = distances_view[i,0] + 1
                for j in range(i+1,N):
                    appo= 0
                    for k in range(L):
                        appo = appo + abs(points[i,k]-points[j,k])
                    if appo <= d_max:
                        ind = int(appo)
                        distances_view[i, ind] = distances_view[i, ind] + 1
                        distances_view[j, ind] = distances_view[j, ind] + 1
                for l in range(1,d_max+1):
                    distances_view[i,l] += distances_view[i,l-1]
    return distances
"""

    # tentative of using an external function
    cdef int inner(pi, pj):
        cdef double appo
        for j in range(i+1,N):
            appo = 0
            for k in range(L):
                appo = appo + abs(pi[k]-pj[k])
            if appo <= d_max:
        return int(appo)

    with nogil, parallel(num_threads=8):
        for i in prange(N, schedule='dynamic'):
            distances_view[i,0] = distances_view[i,0] + 1
            ind = inner(points[i],points[j])
            distances_view[i, ind] = distances_view[i, ind + 1
            distances_view[j, ind] = distances_view[j, ind + 1
            for l in range(1,d_max+1):
                distances_view[i,l] += distances_view[i,l-1]


    # usage of memory view also for appo and ind
    if period is None:
        with nogil, parallel(num_threads=8):
            for i in prange(N, schedule='dynamic'):
                distances_view[i,0] = distances_view[i,0] + 1
                for j in range(i+1,N):
                    appo = 0
                    for k in range(L):
                        appo = appo + abs(points[i,k]-points[j,k])
                    if appov[i] <= d_max:
                        indv[i] = int(appov[i])
                        distances_view[i, indv[i]] = distances_view[i, indv[i]] + 1
                        distances_view[j, indv[i]] = distances_view[j, indv[i]] + 1
                for l in range(1,d_max+1):
                    distances_view[i,l] += distances_view[i,l-1]

    else:       # with period, to be fixed after the normal one
        with nogil, parallel(num_threads=16):
            for i in prange(N, schedule='dynamic'):
                distances_view[i, 0] += 1
                for j in range(i+1, N):
                    appo = 0
                    for k in range(L):
                        d = points[i, k] - points[j, k]
                        appo = appo + fabs(d - nearbyint(d/period[k])*period[k])
                    if appo <= d_max:
                        ind = int(appo)
                        distances_view[i, ind] = distances_view[i, ind] + 1
                        distances_view[j, ind] = distances_view[j, ind] + 1
                for l in range(1, d_max + 1):
                    distances_view[i, l] += distances_view[i, l - 1]
"""

