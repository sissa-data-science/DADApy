import time

import cython
import numpy as np

cimport numpy as np
from cython.parallel cimport prange

DTYPE = np.int_
floatTYPE = np.float_
boolTYPE = np.bool_

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

from libc.math cimport exp, fabs, nearbyint, sqrt  # c FUNCTIONS FASTER THAN NUMPY

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_ind(np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                     np.ndarray[DTYPE_t, ndim = 1] kstar):
    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t kstar_max = np.max(kstar)
    cdef DTYPE_t nspar = kstar.sum() - N
    cdef np.ndarray[DTYPE_t, ndim = 2] nind_list = np.ndarray((nspar, 2), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] nind_iptr = np.ndarray(shape=(N + 1,), dtype=DTYPE)

    cdef DTYPE_t i, j, k, ind_spar, ki

    ind_spar = 0
    for i in range(N):
        nind_iptr[i] = ind_spar
        ki = kstar[i] - 1
        for k in range(ki):
            j = dist_indices[i, k + 1]
            #nind_mat[i,j] = ind_spar
            nind_list[ind_spar, 0] = i
            nind_list[ind_spar, 1] = j
            ind_spar += 1
    nind_iptr[N] = nspar
    assert (ind_spar == nspar)

    #    return nind_list, nind_mat
    return nind_list, nind_iptr

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_ind_parallel(np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                              np.ndarray[DTYPE_t, ndim = 1] kstar,
                              DTYPE_t n_jobs):
    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t nspar = kstar.sum() - N
    cdef np.ndarray[DTYPE_t, ndim = 2] nind_list = np.ndarray((nspar, 2), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] nind_iptr = np.ndarray(shape=(N + 1,), dtype=DTYPE)

    cdef DTYPE_t i, j, k, ki, ind_spar, row_start
    cdef DTYPE_t[::1] kstar_v = kstar
    cdef DTYPE_t[:, ::1] dist_indices_v = dist_indices
    cdef DTYPE_t[:, ::1] nind_list_v = nind_list
    cdef DTYPE_t[::1] nind_iptr_v = nind_iptr

    ind_spar = 0
    for i in range(N):
        nind_iptr_v[i] = ind_spar
        ind_spar += kstar_v[i] - 1
    nind_iptr_v[N] = nspar

    assert (ind_spar == nspar)

    with nogil:
        for i in prange(N, schedule='static', num_threads=n_jobs):
            row_start = nind_iptr_v[i]
            ki = kstar_v[i] - 1
            for k in range(ki):
                j = dist_indices_v[i, k + 1]
                nind_list_v[row_start + k, 0] = i
                nind_list_v[row_start + k, 1] = j

    return nind_list, nind_iptr

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_distances_array(   np.ndarray[floatTYPE_t, ndim = 2] distances,
                                    np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                    np.ndarray[DTYPE_t, ndim = 1] kstar):
    cdef DTYPE_t N = len(kstar)
    cdef DTYPE_t nspar = kstar.sum() - N
    cdef np.ndarray[floatTYPE_t, ndim = 1] distarray = np.ndarray((nspar,), dtype=floatTYPE)

    cdef DTYPE_t i, j, ind_spar

    ind_spar = 0
    for i in range(N):
        for j in range(1,kstar[i]):
            distarray[ind_spar] =  distances[i,j]
            ind_spar += 1

    assert (ind_spar == nspar)

    return distarray

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_distances_array_parallel(np.ndarray[floatTYPE_t, ndim = 2] distances,
                                          np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                          np.ndarray[DTYPE_t, ndim = 1] kstar,
                                          DTYPE_t n_jobs):
    cdef DTYPE_t N = len(kstar)
    cdef DTYPE_t nspar = kstar.sum() - N
    cdef np.ndarray[floatTYPE_t, ndim = 1] distarray = np.ndarray((nspar,), dtype=floatTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] starts = np.ndarray((N,), dtype=DTYPE)

    cdef DTYPE_t i, j, ind_spar
    cdef DTYPE_t[::1] kstar_v = kstar
    cdef floatTYPE_t[:, ::1] distances_v = distances
    cdef floatTYPE_t[::1] distarray_v = distarray
    cdef DTYPE_t[::1] starts_v = starts
    cdef DTYPE_t start_i, ki

    ind_spar = 0
    for i in range(N):
        starts_v[i] = ind_spar
        ind_spar += kstar_v[i] - 1

    assert (ind_spar == nspar)

    with nogil:
        for i in prange(N, schedule='static', num_threads=n_jobs):
            start_i = starts_v[i]
            ki = kstar_v[i]
            for j in range(1, ki):
                distarray_v[start_i + j - 1] = distances_v[i, j]

    return distarray

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_vector_diffs(np.ndarray[floatTYPE_t, ndim = 2] X,
                              np.ndarray[DTYPE_t, ndim = 2] nind_list):
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims), dtype=floatTYPE)

    cdef DTYPE_t i, j, ind_spar, dim

    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]
        for dim in range(dims):
            vector_diffs[ind_spar, dim] = X[j, dim] - X[i, dim]

    return vector_diffs

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_vector_diffs_parallel(np.ndarray[floatTYPE_t, ndim = 2] X,
                                       np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                       DTYPE_t n_jobs):
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims), dtype=floatTYPE)

    cdef DTYPE_t i, j, ind_spar, dim
    cdef floatTYPE_t[:, ::1] Xv = X
    cdef DTYPE_t[:, ::1] nind_v = nind_list
    cdef floatTYPE_t[:, ::1] vector_diffs_v = vector_diffs

    with nogil:
        for ind_spar in prange(nspar, schedule='static', num_threads=n_jobs):
            i = nind_v[ind_spar, 0]
            j = nind_v[ind_spar, 1]
            for dim in range(dims):
                vector_diffs_v[ind_spar, dim] = Xv[j, dim] - Xv[i, dim]

    return vector_diffs

# ----------------------------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_vector_diffs_periodic(np.ndarray[floatTYPE_t, ndim = 2] X,
                              np.ndarray[DTYPE_t, ndim = 2] nind_list,
                              np.ndarray[floatTYPE_t, ndim = 1] period):
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims), dtype=floatTYPE)

    cdef DTYPE_t i, j, ind_spar, dim
    cdef floatTYPE_t temp

    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]
        for dim in range(dims):
            temp = X[j, dim] - X[i, dim]
            if temp > period[dim]/2:
                temp -= period[dim]
            if temp < -period[dim]/2:
                temp += period[dim] 
            vector_diffs[ind_spar, dim] = temp

    return vector_diffs

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_vector_diffs_periodic_parallel(np.ndarray[floatTYPE_t, ndim = 2] X,
                                                np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                                np.ndarray[floatTYPE_t, ndim = 1] period,
                                                DTYPE_t n_jobs):
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims), dtype=floatTYPE)

    cdef DTYPE_t i, j, ind_spar, dim
    cdef floatTYPE_t temp, half_period, period_dim
    cdef floatTYPE_t[:, ::1] Xv = X
    cdef DTYPE_t[:, ::1] nind_v = nind_list
    cdef floatTYPE_t[::1] period_v = period
    cdef floatTYPE_t[:, ::1] vector_diffs_v = vector_diffs

    with nogil:
        for ind_spar in prange(nspar, schedule='static', num_threads=n_jobs):
            i = nind_v[ind_spar, 0]
            j = nind_v[ind_spar, 1]
            for dim in range(dims):
                temp = Xv[j, dim] - Xv[i, dim]
                period_dim = period_v[dim]
                half_period = period_dim / 2.0
                if temp > half_period:
                    temp = temp - period_dim
                if temp < -half_period:
                    temp = temp + period_dim
                vector_diffs_v[ind_spar, dim] = temp

    return vector_diffs

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def return_common_neighs(np.ndarray[DTYPE_t, ndim = 1] kstar,
                         np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                         np.ndarray[DTYPE_t, ndim = 2] nind_list):

    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t maxk = kstar.shape[1]
    cdef DTYPE_t nspar = nind_list.shape[0]

    cdef DTYPE_t i, j, ind_spar, count, kstar_i, kstar_j, idx, idx2, val_i, val_j

    cdef np.ndarray[DTYPE_t, ndim=1] common_neighs_array = np.zeros(nspar, dtype=DTYPE)

    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]

        kstar_i = kstar[i]
        kstar_j = kstar[j]

        count = 0
        idx = 0
        idx2 = 0

        for idx in range(kstar_i):
            val_i = dist_indices[i, idx]
            for idx2 in range(kstar_j):
                val_j = dist_indices[j, idx2]
                if val_i == val_j:
                    count += 1
                    break #no point in checking further

        common_neighs_array[ind_spar] = count

    return common_neighs_array
# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def return_common_neighs_parallel(np.ndarray[DTYPE_t, ndim = 1] kstar,
                                  np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                  np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                  DTYPE_t n_jobs):

    cdef DTYPE_t nspar = nind_list.shape[0]

    cdef DTYPE_t i, j, ind_spar, count, kstar_i, kstar_j, idx, idx2, val_i, val_j
    cdef np.ndarray[DTYPE_t, ndim=1] common_neighs_array = np.zeros(nspar, dtype=DTYPE)

    cdef DTYPE_t[::1] kstar_v = kstar
    cdef DTYPE_t[:, ::1] dist_indices_v = dist_indices
    cdef DTYPE_t[:, ::1] nind_v = nind_list
    cdef DTYPE_t[::1] common_v = common_neighs_array

    with nogil:
        for ind_spar in prange(nspar, schedule='static', num_threads=n_jobs):
            i = nind_v[ind_spar, 0]
            j = nind_v[ind_spar, 1]

            kstar_i = kstar_v[i]
            kstar_j = kstar_v[j]

            count = 0

            for idx in range(kstar_i):
                val_i = dist_indices_v[i, idx]
                for idx2 in range(kstar_j):
                    val_j = dist_indices_v[j, idx2]
                    if val_i == val_j:
                        count = count + 1
                        break

            common_v[ind_spar] = count

    return common_neighs_array

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_common_neighs_comp_mat(np.ndarray[DTYPE_t, ndim = 1] kstar,
                         np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                         np.ndarray[DTYPE_t, ndim = 2] nind_list):
    
    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t maxk = kstar.shape[1]
    cdef DTYPE_t nspar = nind_list.shape[0]

    cdef DTYPE_t i, j, ind_spar, count, kstar_i, kstar_j, idx, idx2, val_i, val_j

    cdef np.ndarray[DTYPE_t, ndim=1] common_neighs_array = np.zeros(nspar, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] common_neighs_mat = np.zeros((N,N), dtype=DTYPE)

    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]
        if common_neighs_mat[j,i] == 0:
            kstar_i = kstar[i]
            kstar_j = kstar[j]

            count = 0
            idx = 0
            idx2 = 0

            for idx in range(kstar_i):
                val_i = dist_indices[i, idx]
                for idx2 in range(kstar_j):
                    val_j = dist_indices[j, idx2]
                    if val_i == val_j:
                        count += 1
                        break #no point in checking further

            common_neighs_mat[i,j] = count
            common_neighs_mat[j,i] = count
            common_neighs_array[ind_spar] = count
        else:
            common_neighs_mat[i,j] = common_neighs_mat[j,i]
            common_neighs_array[ind_spar] = common_neighs_mat[j,i]

    return common_neighs_array, common_neighs_mat

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def return_cross_common_neighs( np.ndarray[DTYPE_t, ndim = 1] kstar,
                                np.ndarray[DTYPE_t, ndim = 1] kstar_test,
                                np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                np.ndarray[DTYPE_t, ndim = 2] cross_dist_indices,
                                np.ndarray[DTYPE_t, ndim = 2] cross_nind_list
                                ):

    cdef DTYPE_t N = kstar_test.shape[0]
    cdef DTYPE_t maxk = kstar_test.shape[1]
    cdef DTYPE_t nspar = cross_nind_list.shape[0]

    cdef DTYPE_t i, j, ind_spar, count, kstar_i, kstar_j, idx, idx2, val_i, val_j

    cdef np.ndarray[DTYPE_t, ndim=1] common_neighs_array = np.zeros(nspar, dtype=DTYPE)

    for ind_spar in range(nspar):
        i = cross_nind_list[ind_spar, 0]
        j = cross_nind_list[ind_spar, 1]

        kstar_i = kstar_test[i]
        kstar_j = kstar[j]

        count = 0
        idx = 0
        idx2 = 0

        for idx in range(kstar_i):
            val_i = cross_dist_indices[i, idx]
            for idx2 in range(kstar_j):
                val_j = dist_indices[j, idx2]
                if val_i == val_j:
                    count += 1
                    break #no point in checking further

        common_neighs_array[ind_spar] = count

    return common_neighs_array

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def return_cross_common_neighs_parallel(np.ndarray[DTYPE_t, ndim = 1] kstar,
                                        np.ndarray[DTYPE_t, ndim = 1] kstar_test,
                                        np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                        np.ndarray[DTYPE_t, ndim = 2] cross_dist_indices,
                                        np.ndarray[DTYPE_t, ndim = 2] cross_nind_list,
                                        DTYPE_t n_jobs):

    cdef DTYPE_t nspar = cross_nind_list.shape[0]

    cdef DTYPE_t i, j, ind_spar, count, kstar_i, kstar_j, idx, idx2, val_i, val_j
    cdef np.ndarray[DTYPE_t, ndim=1] common_neighs_array = np.zeros(nspar, dtype=DTYPE)

    cdef DTYPE_t[::1] kstar_v = kstar
    cdef DTYPE_t[::1] kstar_test_v = kstar_test
    cdef DTYPE_t[:, ::1] dist_indices_v = dist_indices
    cdef DTYPE_t[:, ::1] cross_dist_indices_v = cross_dist_indices
    cdef DTYPE_t[:, ::1] cross_nind_v = cross_nind_list
    cdef DTYPE_t[::1] common_v = common_neighs_array

    with nogil:
        for ind_spar in prange(nspar, schedule='static', num_threads=n_jobs):
            i = cross_nind_v[ind_spar, 0]
            j = cross_nind_v[ind_spar, 1]

            kstar_i = kstar_test_v[i]
            kstar_j = kstar_v[j]

            count = 0

            for idx in range(kstar_i):
                val_i = cross_dist_indices_v[i, idx]
                for idx2 in range(kstar_j):
                    val_j = dist_indices_v[j, idx2]
                    if val_i == val_j:
                        count = count + 1
                        break

            common_v[ind_spar] = count

    return common_neighs_array

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def return_deltaFs_and_var_from_grads(  np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                        np.ndarray[floatTYPE_t, ndim = 2] grads,
                                        np.ndarray[floatTYPE_t, ndim = 3] grads_covmat,
                                        np.ndarray[floatTYPE_t, ndim = 2] neigh_vector_diffs,
                                        np.ndarray[floatTYPE_t, ndim=1] pearson_array                                        
):
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef DTYPE_t dims = neigh_vector_diffs.shape[1]
    cdef DTYPE_t N = grads.shape[0]

    cdef DTYPE_t ind_spar, dim, dim2, i, j
    cdef floatTYPE_t grad_dot, vari, varj, dx_dim, tmpi, tmpj

    cdef np.ndarray[floatTYPE_t, ndim=1] Fij_array = np.zeros(nspar, dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim=1] Fij_var_array = np.zeros(nspar, dtype=floatTYPE)
        

    if neigh_vector_diffs.shape[0] != nspar:
        raise ValueError("nind_list and neigh_vector_diffs must have the same length")
    if pearson_array.shape[0] != nspar:
        raise ValueError("pearson_array length must match nind_list length")
    if grads.shape[1] != dims:
        raise ValueError("grads and neigh_vector_diffs must have the same dimension")
    if grads_covmat.shape[0] != N:
        raise ValueError("grads_covmat and grads must have the same number of points")
    if grads_covmat.shape[1] != dims or grads_covmat.shape[2] != dims:
        raise ValueError("grads_covmat shape must be (N, dims, dims)")
    
    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]

        grad_dot = 0.
        vari = 0.
        varj = 0.

        for dim in range(dims):
            grad_dot += (grads[i, dim] + grads[j, dim]) * neigh_vector_diffs[ind_spar, dim]

        for dim in range(dims):
            dx_dim = neigh_vector_diffs[ind_spar, dim]
            tmpi = 0.
            tmpj = 0.
            for dim2 in range(dims):
                tmpi += grads_covmat[i, dim, dim2] * neigh_vector_diffs[ind_spar, dim2]
                tmpj += grads_covmat[j, dim, dim2] * neigh_vector_diffs[ind_spar, dim2]
            vari += dx_dim * tmpi
            varj += dx_dim * tmpj

        Fij_array[ind_spar] = 0.5 * grad_dot
        Fij_var_array[ind_spar] = 0.25 * (
            vari + varj + 2. * pearson_array[ind_spar] * sqrt(vari * varj)
        )

    return Fij_array, Fij_var_array

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def return_deltaFs_and_var_from_grads_parallel(np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                               np.ndarray[floatTYPE_t, ndim = 2] grads,
                                               np.ndarray[floatTYPE_t, ndim = 3] grads_covmat,
                                               np.ndarray[floatTYPE_t, ndim = 2] neigh_vector_diffs,
                                               np.ndarray[floatTYPE_t, ndim=1] pearson_array,
                                               DTYPE_t n_jobs):
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef DTYPE_t dims = neigh_vector_diffs.shape[1]
    cdef DTYPE_t N = grads.shape[0]

    cdef DTYPE_t ind_spar, dim, dim2, i, j
    cdef floatTYPE_t grad_dot, vari, varj, dx_dim, tmpi, tmpj

    cdef np.ndarray[floatTYPE_t, ndim=1] Fij_array = np.zeros(nspar, dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim=1] Fij_var_array = np.zeros(nspar, dtype=floatTYPE)

    cdef DTYPE_t[:, ::1] nind_v = nind_list
    cdef floatTYPE_t[:, ::1] grads_v = grads
    cdef floatTYPE_t[:, :, ::1] grads_cov_v = grads_covmat
    cdef floatTYPE_t[:, ::1] neigh_v = neigh_vector_diffs
    cdef floatTYPE_t[::1] pearson_v = pearson_array
    cdef floatTYPE_t[::1] fij_v = Fij_array
    cdef floatTYPE_t[::1] fij_var_v = Fij_var_array

    if neigh_vector_diffs.shape[0] != nspar:
        raise ValueError("nind_list and neigh_vector_diffs must have the same length")
    if pearson_array.shape[0] != nspar:
        raise ValueError("pearson_array length must match nind_list length")
    if grads.shape[1] != dims:
        raise ValueError("grads and neigh_vector_diffs must have the same dimension")
    if grads_covmat.shape[0] != N:
        raise ValueError("grads_covmat and grads must have the same number of points")
    if grads_covmat.shape[1] != dims or grads_covmat.shape[2] != dims:
        raise ValueError("grads_covmat shape must be (N, dims, dims)")

    with nogil:
        for ind_spar in prange(nspar, schedule='static', num_threads=n_jobs):
            i = nind_v[ind_spar, 0]
            j = nind_v[ind_spar, 1]

            grad_dot = 0.
            vari = 0.
            varj = 0.

            for dim in range(dims):
                grad_dot = grad_dot + (grads_v[i, dim] + grads_v[j, dim]) * neigh_v[ind_spar, dim]

            for dim in range(dims):
                dx_dim = neigh_v[ind_spar, dim]
                tmpi = 0.
                tmpj = 0.
                for dim2 in range(dims):
                    tmpi = tmpi + grads_cov_v[i, dim, dim2] * neigh_v[ind_spar, dim2]
                    tmpj = tmpj + grads_cov_v[j, dim, dim2] * neigh_v[ind_spar, dim2]
                vari = vari + dx_dim * tmpi
                varj = varj + dx_dim * tmpj

            fij_v[ind_spar] = 0.5 * grad_dot
            fij_var_v[ind_spar] = 0.25 * (
                vari + varj + 2. * pearson_v[ind_spar] * sqrt(vari * varj)
            )

    return Fij_array, Fij_var_array

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def return_deltaFs_from_grads(  np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                np.ndarray[floatTYPE_t, ndim = 2] grads,
                                np.ndarray[floatTYPE_t, ndim = 2] neigh_vector_diffs
):
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef DTYPE_t dims = neigh_vector_diffs.shape[1]

    cdef DTYPE_t ind_spar, dim, i, j
    cdef floatTYPE_t grad_dot

    cdef np.ndarray[floatTYPE_t, ndim=1] Fij_array = np.zeros(nspar, dtype=floatTYPE)
        

    if neigh_vector_diffs.shape[0] != nspar:
        raise ValueError("nind_list and neigh_vector_diffs must have the same length")
    if grads.shape[1] != dims:
        raise ValueError("grads and neigh_vector_diffs must have the same dimension")
    
    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]

        grad_dot = 0.

        for dim in range(dims):
            grad_dot += (grads[i, dim] + grads[j, dim]) * neigh_vector_diffs[ind_spar, dim]

        Fij_array[ind_spar] = 0.5 * grad_dot

    return Fij_array

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def return_deltaFs_from_grads_parallel(np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                       np.ndarray[floatTYPE_t, ndim = 2] grads,
                                       np.ndarray[floatTYPE_t, ndim = 2] neigh_vector_diffs,
                                       DTYPE_t n_jobs):
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef DTYPE_t dims = neigh_vector_diffs.shape[1]

    cdef DTYPE_t ind_spar, dim, i, j
    cdef floatTYPE_t grad_dot

    cdef np.ndarray[floatTYPE_t, ndim=1] Fij_array = np.zeros(nspar, dtype=floatTYPE)

    cdef DTYPE_t[:, ::1] nind_v = nind_list
    cdef floatTYPE_t[:, ::1] grads_v = grads
    cdef floatTYPE_t[:, ::1] neigh_v = neigh_vector_diffs
    cdef floatTYPE_t[::1] fij_v = Fij_array

    if neigh_vector_diffs.shape[0] != nspar:
        raise ValueError("nind_list and neigh_vector_diffs must have the same length")
    if grads.shape[1] != dims:
        raise ValueError("grads and neigh_vector_diffs must have the same dimension")

    with nogil:
        for ind_spar in prange(nspar, schedule='static', num_threads=n_jobs):
            i = nind_v[ind_spar, 0]
            j = nind_v[ind_spar, 1]

            grad_dot = 0.

            for dim in range(dims):
                grad_dot = grad_dot + (grads_v[i, dim] + grads_v[j, dim]) * neigh_v[ind_spar, dim]

            fij_v[ind_spar] = 0.5 * grad_dot

    return Fij_array

# ----------------------------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.cdivision(True)
def return_grads_and_var_from_coords(  np.ndarray[floatTYPE_t, ndim = 2] X,
                                        np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                        np.ndarray[DTYPE_t, ndim = 1] kstar,
                                        floatTYPE_t id_selected):
# NOT USED AT THE MOMENT

    cdef DTYPE_t N = X.shape[0]
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads_var = np.zeros((N, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 1] rk_sq_arr = np.zeros((N,), dtype=floatTYPE)
    
    cdef DTYPE_t i, j, dim, ki, dim2
    cdef DTYPE_t ind_j
    cdef floatTYPE_t rk_sq, kifloat
    cdef floatTYPE_t dp2 = id_selected + 2.

    for i in range(N):
        ki = kstar[i]-1

        kifloat = float(ki)

        rk_sq = 0.
        for dim in range(dims):
            rk_sq += (X[dist_indices[i, ki+1], dim] - X[i, dim])**2

        # compute gradients and variance of gradients together
        for dim in range(dims):
            for j in range(ki):
                ind_j = dist_indices[i, j+1]

                grads[i, dim] += (X[ind_j, dim] - X[i, dim])
                grads_var[i, dim] += (X[ind_j, dim] - X[i, dim]) * (X[ind_j, dim] - X[i, dim])

            grads[i, dim] = grads[i, dim] / kifloat * dp2/rk_sq

            grads_var[i, dim] = grads_var[i, dim] / kifloat / kifloat * dp2/rk_sq * dp2/rk_sq \
                              - grads[i, dim]*grads[i, dim] / kifloat

    return grads, grads_var

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_grads_and_covmat_from_coords(   np.ndarray[floatTYPE_t, ndim = 2] X,
                                            np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                            np.ndarray[DTYPE_t, ndim = 1] kstar,
                                            floatTYPE_t id_selected):
# NOT USED AT THE MOMENT

    cdef DTYPE_t N = X.shape[0]
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 3] grads_covmat = np.zeros((N, dims, dims), dtype=floatTYPE)

    cdef DTYPE_t i, j, dim, ki, dim2
    cdef DTYPE_t ind_j
    cdef floatTYPE_t rk_sq, kifloat
    cdef floatTYPE_t dp2 = id_selected + 2.

    for i in range(N):
        ki = kstar[i]-1

        kifloat = float(ki)

        rk_sq = 0.
        for dim in range(dims):
            rk_sq += (X[dist_indices[i, ki+1], dim] - X[i, dim])**2

        # compute gradients
        for dim in range(dims):
            for j in range(ki):
                ind_j = dist_indices[i, j+1]

                grads[i, dim] += (X[ind_j, dim] - X[i, dim])

            grads[i, dim] = grads[i, dim] / kifloat * dp2/rk_sq

        # compute covariance matrix of gradients
        for dim in range(dims):
            for dim2 in range(dims):
                for j in range(ki):
                    ind_j = dist_indices[i, j+1]

                    grads_covmat[i, dim, dim2] += (X[ind_j, dim] - X[i, dim]) * (X[ind_j, dim2] - X[i, dim2])

                grads_covmat[i, dim, dim2] = grads_covmat[i, dim, dim2] / kifloat / kifloat * dp2/rk_sq * dp2/rk_sq \
                                  - grads[i, dim]*grads[i, dim2] / kifloat

    return grads, grads_covmat

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_grads_and_var_from_nnvecdiffs(   np.ndarray[floatTYPE_t, ndim = 2] neigh_vector_diffs,
                                            np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                            np.ndarray[DTYPE_t, ndim = 1] nind_iptr,
                                            np.ndarray[DTYPE_t, ndim = 1] kstar,
                                            floatTYPE_t id_selected):

    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t dims = neigh_vector_diffs.shape[1]
    cdef DTYPE_t kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads_var = np.zeros((N, dims), dtype=floatTYPE)
    
    cdef DTYPE_t i, j, dim, ki, dim2
    cdef DTYPE_t ind_j
    cdef floatTYPE_t rk_sq, kifloat
    cdef floatTYPE_t dp2 = id_selected + 2.


    for i in range(N):
        ki = kstar[i]-1

        kifloat = float(ki)

        rk_sq = 0.
        for dim in range(dims):
            rk_sq += (neigh_vector_diffs[nind_iptr[i+1]-1,dim])**2

        # compute gradients and variance of gradients together
        for dim in range(dims):
            for j in range(ki):
                ind_j = nind_iptr[i]+j

                grads[i, dim] += neigh_vector_diffs[ind_j,dim]
                grads_var[i, dim] += neigh_vector_diffs[ind_j,dim]*neigh_vector_diffs[ind_j,dim]

            grads[i, dim] = grads[i, dim] / kifloat * dp2/rk_sq

            grads_var[i, dim] = grads_var[i, dim] / kifloat / kifloat * dp2/rk_sq * dp2/rk_sq \
                              - grads[i, dim]*grads[i, dim] / kifloat

    return grads, grads_var

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_grads_and_var_from_nnvecdiffs_parallel(np.ndarray[floatTYPE_t, ndim = 2] neigh_vector_diffs,
                                                   np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                                   np.ndarray[DTYPE_t, ndim = 1] nind_iptr,
                                                   np.ndarray[DTYPE_t, ndim = 1] kstar,
                                                   floatTYPE_t id_selected,
                                                   DTYPE_t n_jobs):

    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t dims = neigh_vector_diffs.shape[1]
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads_var = np.zeros((N, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 1] rk_sq_arr = np.zeros((N,), dtype=floatTYPE)

    cdef DTYPE_t i, j, dim, ki
    cdef DTYPE_t ind_j
    cdef floatTYPE_t rk_sq, kifloat, scale, diff
    cdef floatTYPE_t dp2 = id_selected + 2.

    cdef floatTYPE_t[:, ::1] neigh_v = neigh_vector_diffs
    cdef DTYPE_t[::1] nind_iptr_v = nind_iptr
    cdef DTYPE_t[::1] kstar_v = kstar
    cdef floatTYPE_t[:, ::1] grads_v = grads
    cdef floatTYPE_t[:, ::1] grads_var_v = grads_var
    cdef floatTYPE_t[::1] rk_sq_arr_v = rk_sq_arr

    for i in range(N):
        rk_sq = 0.
        for dim in range(dims):
            diff = neigh_v[nind_iptr_v[i + 1] - 1, dim]
            rk_sq += diff * diff
        rk_sq_arr_v[i] = rk_sq

    with nogil:
        for i in prange(N, schedule='static', num_threads=n_jobs):
            ki = kstar_v[i] - 1
            if ki <= 0:
                continue

            kifloat = <floatTYPE_t>ki
            rk_sq = rk_sq_arr_v[i]
            scale = dp2 / rk_sq

            for dim in range(dims):
                for j in range(ki):
                    ind_j = nind_iptr_v[i] + j
                    diff = neigh_v[ind_j, dim]
                    grads_v[i, dim] += diff
                    grads_var_v[i, dim] += diff * diff

                grads_v[i, dim] = grads_v[i, dim] / kifloat * scale
                grads_var_v[i, dim] = (
                    grads_var_v[i, dim] / (kifloat * kifloat) * scale * scale
                    - grads_v[i, dim] * grads_v[i, dim] / kifloat
                )

    return grads, grads_var

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_grads_and_covmat_from_nnvecdiffs(np.ndarray[floatTYPE_t, ndim = 2] neigh_vector_diffs,
                                            np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                            np.ndarray[DTYPE_t, ndim = 1] nind_iptr,
                                            np.ndarray[DTYPE_t, ndim = 1] kstar,
                                            floatTYPE_t id_selected):

    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t dims = neigh_vector_diffs.shape[1]
    cdef DTYPE_t kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 3] grads_covmat = np.zeros((N, dims, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 1] rk_sq_arr = np.zeros((N,), dtype=floatTYPE)

    cdef DTYPE_t i, j, dim, ki, dim2
    cdef DTYPE_t ind_j
    cdef floatTYPE_t rk_sq, kifloat
    cdef floatTYPE_t dp2 = id_selected + 2.

    for i in range(N):
        ki = kstar[i]-1

        kifloat = float(ki)

        rk_sq = 0.
        for dim in range(dims):
            rk_sq += (neigh_vector_diffs[nind_iptr[i+1]-1,dim])**2

        # compute gradients
        for dim in range(dims):
            for j in range(ki):
                ind_j = nind_iptr[i]+j

                grads[i, dim] += neigh_vector_diffs[ind_j,dim]

            grads[i, dim] = grads[i, dim] / kifloat * dp2/rk_sq

        # compute covariance matrix of gradients
        for dim in range(dims):
            for dim2 in range(dims):
                for j in range(ki):
                    ind_j = nind_iptr[i]+j

                    grads_covmat[i, dim, dim2] += neigh_vector_diffs[ind_j,dim]*neigh_vector_diffs[ind_j,dim2]

                grads_covmat[i, dim, dim2] = grads_covmat[i, dim, dim2] / kifloat / kifloat * dp2/rk_sq * dp2/rk_sq \
                                  - grads[i, dim]*grads[i, dim2] / kifloat

    return grads, grads_covmat

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_grads_and_covmat_from_nnvecdiffs_parallel(np.ndarray[floatTYPE_t, ndim = 2] neigh_vector_diffs,
                                                     np.ndarray[DTYPE_t, ndim = 2] nind_list,
                                                     np.ndarray[DTYPE_t, ndim = 1] nind_iptr,
                                                     np.ndarray[DTYPE_t, ndim = 1] kstar,
                                                     floatTYPE_t id_selected,
                                                     DTYPE_t n_jobs):

    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t dims = neigh_vector_diffs.shape[1]
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 3] grads_covmat = np.zeros((N, dims, dims), dtype=floatTYPE)
    cdef np.ndarray[floatTYPE_t, ndim = 1] rk_sq_arr = np.zeros((N,), dtype=floatTYPE)

    cdef DTYPE_t i, j, dim, dim2, ki
    cdef DTYPE_t ind_j
    cdef floatTYPE_t rk_sq, kifloat, scale, diff_dim, diff_dim2
    cdef floatTYPE_t dp2 = id_selected + 2.

    cdef floatTYPE_t[:, ::1] neigh_v = neigh_vector_diffs
    cdef DTYPE_t[::1] nind_iptr_v = nind_iptr
    cdef DTYPE_t[::1] kstar_v = kstar
    cdef floatTYPE_t[:, ::1] grads_v = grads
    cdef floatTYPE_t[:, :, ::1] grads_covmat_v = grads_covmat
    cdef floatTYPE_t[::1] rk_sq_arr_v = rk_sq_arr

    for i in range(N):
        rk_sq = 0.
        for dim in range(dims):
            diff_dim = neigh_v[nind_iptr_v[i + 1] - 1, dim]
            rk_sq += diff_dim * diff_dim
        rk_sq_arr_v[i] = rk_sq

    with nogil:
        for i in prange(N, schedule='static', num_threads=n_jobs):
            ki = kstar_v[i] - 1
            if ki <= 0:
                continue

            kifloat = <floatTYPE_t>ki
            rk_sq = rk_sq_arr_v[i]
            scale = dp2 / rk_sq

            for dim in range(dims):
                for j in range(ki):
                    ind_j = nind_iptr_v[i] + j
                    grads_v[i, dim] += neigh_v[ind_j, dim]

                grads_v[i, dim] = grads_v[i, dim] / kifloat * scale

            for dim in range(dims):
                for dim2 in range(dims):
                    for j in range(ki):
                        ind_j = nind_iptr_v[i] + j
                        diff_dim = neigh_v[ind_j, dim]
                        diff_dim2 = neigh_v[ind_j, dim2]
                        grads_covmat_v[i, dim, dim2] += diff_dim * diff_dim2

                    grads_covmat_v[i, dim, dim2] = (
                        grads_covmat_v[i, dim, dim2] / (kifloat * kifloat) * scale * scale
                        - grads_v[i, dim] * grads_v[i, dim2] / kifloat
                    )

    return grads, grads_covmat

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_diag_inv_deltaFs_cross_covariance_LSDI(long[:,:] nind_list,      # nspar x 2
                                        double[:,:] p,                  # neigh_similarity_index matrix (NxN)
                                        double[:] Fij_var_array,
                                        double[:] seps0,
                                        double[:] seps1
                                        ):
    cdef int nspar = nind_list.shape[0]

    inv_Gamma_nonview   = np.zeros(nspar, dtype=floatTYPE)       # inverse of diagonal of Gamma matrix
    cdef double[::1] inv_Gamma = inv_Gamma_nonview
    
    #support
    denom_nonview   = np.zeros(nspar, dtype=floatTYPE)
    cdef double[::1] denom = denom_nonview

    cdef double gamma, ptot, sgn
    cdef int i,j,l,m,a,b  

    for a in range(nspar):
        i = nind_list[a, 0]
        j = nind_list[a, 1]
        inv_Gamma[a] = Fij_var_array[a]
        denom[a] += Fij_var_array[a]*Fij_var_array[a]
        for b in range(a+1, nspar):
            l = nind_list[b, 0]
            m = nind_list[b, 1]
            gamma = 0
            ptot = 0
            if p[i,l] != 0:
                ptot += 1
                gamma += p[i,l]*seps0[a]*seps0[b]
            if p[i,m] != 0:
                gamma += p[i,m]*seps0[a]*seps1[b]
            if p[j,l] != 0:
                ptot += 1
                gamma += p[j,l]*seps1[a]*seps0[b]
            if p[j,m] != 0:
                gamma += p[j,m]*seps1[a]*seps1[b]
            if ptot != 0:
                denom[a] += gamma * gamma / 16.
                denom[b] += gamma * gamma / 16.
        
    for a in range(nspar):
        inv_Gamma[a] /= denom[a]

    #return Gamma, inv_Gamma
    return np.asarray(inv_Gamma)

# ----------------------------------------------------------------------------------------------
