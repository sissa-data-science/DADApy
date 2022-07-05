import cython
import numpy as np

cimport numpy as np
from cython.parallel cimport parallel, prange
from libc.math cimport nearbyint
from libc.stdlib cimport abs

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

    cdef long double operation = 0
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
    cdef int i, j, k, d, temp, appo
    cdef np.ndarray[DTYPE_t, ndim = 2] distances = np.zeros((N, d_max + 1), dtype=int)

    if period is None:
        for i in range(N):
            distances[i,0] += 1
            for j in range(i+1,N):
                appo = 0
                for k in range(L):
                    appo += (abs(points[i,k]-points[j,k]))
                if appo <= d_max:
                    distances[i, appo] += 1
                    distances[j, appo] += 1
            for k in range(1,d_max+1):
                distances[i,k] += distances[i,k-1]

    else:
        for i in range(N):
            distances[i, 0] += 1
            for j in range(i+1, N):
                appo = 0
                for k in range(L):
                    d = points[i, k] - points[j, k]
                    temp = int(nearbyint(d/period[k]))
                    appo += (abs(d - temp)*period[k])
                if appo <= d_max:
                    distances[i, appo] += 1
                    distances[j, appo] += 1
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

    cdef int N = len(points)
    cdef int L = len(points[0])
    cdef int i, j, k, d, temp, appo
    cdef np.ndarray[DTYPE_t, ndim = 2] distances = np.zeros((N, d_max + 1), dtype=int)
    cdef Py_ssize_t [:,:] distances_view = distances

    if period is None:
        with nogil, parallel(num_threads=16):
            for i in prange(N, schedule='dynamic'):
                distances_view[i,0] += 1
                for j in range(i+1,N):
                    appo = 0
                    for k in range(L):
                        appo = appo + (abs(points[i,k]-points[j,k]))
                    if appo <= d_max:
                        distances_view[i, appo] = distances_view[i, appo] + 1
                        distances_view[j, appo] = distances_view[j, appo] + 1
                for k in range(1,d_max+1):
                    distances_view[i,k] += distances_view[i,k-1]

    else:
        with nogil, parallel(num_threads=16):
            for i in prange(N, schedule='dynamic'):
                distances_view[i, 0] += 1
                for j in range(i+1, N):
                    appo = 0
                    for k in range(L):
                        d = points[i, k] - points[j, k]
                        temp = int(nearbyint(d/period[k]))
                        appo = appo + (abs(d - temp)*period[k])
                    if appo <= d_max:
                        distances_view[i, appo] = distances_view[i, appo] + 1
                        distances_view[j, appo] = distances_view[j, appo] + 1
                for k in range(1, d_max + 1):
                    distances_view[i, k] += distances_view[i, k - 1]

    return distances

