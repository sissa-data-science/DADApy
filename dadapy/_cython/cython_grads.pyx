import time

import cython
import numpy as np

cimport numpy as np

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
@cython.wraparound(False)
def return_common_neighs(np.ndarray[DTYPE_t, ndim = 1] kstar,
                         np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                         np.ndarray[DTYPE_t, ndim = 2] nind_list):

    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t maxk = kstar.shape[1]
    cdef DTYPE_t nspar = nind_list.shape[0]

    cdef DTYPE_t i, j, ind_spar, count, kstar_i, kstar_j, idx, idx2, val_i, val_j

    cdef np.ndarray[DTYPE_t, ndim=1] common_neighs_array = np.zeros(nspar, dtype=np.int_)
    cdef np.ndarray[DTYPE_t, ndim=2] sorted_dist_indices = np.zeros((N, maxk), dtype=np.int_)

    sorted_dist_indices = np.sort(dist_indices,axis=1)

    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]

        kstar_i = kstar[i]
        kstar_j = kstar[j]

        count = 0
        idx = 0
        idx2 = 0

        # Two-pointer intersection if sorted
        while idx < kstar_i and idx2 < kstar_j:
            val_i = sorted_dist_indices[i, idx]
            val_j = sorted_dist_indices[j, idx2]
            if val_i < val_j:
                idx += 1
            elif val_i > val_j:
                idx2 += 1
            else:
                count += 1
                idx += 1
                idx2 += 1

        common_neighs_array[ind_spar] = count

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

    cdef np.ndarray[DTYPE_t, ndim=1] common_neighs_array = np.zeros(nspar, dtype=np.int_)
    cdef np.ndarray[DTYPE_t, ndim=2] common_neighs_mat = np.zeros((N,N), dtype=np.int_)
    cdef np.ndarray[DTYPE_t, ndim=2] sorted_dist_indices = np.zeros((N, maxk), dtype=np.int_)

    sorted_dist_indices = np.sort(dist_indices,axis=1)

    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]
        if common_neighs_mat[j,i] == 0:
            kstar_i = kstar[i]
            kstar_j = kstar[j]

            count = 0
            idx = 0
            idx2 = 0

            # Two-pointer intersection if sorted
            while idx < kstar_i and idx2 < kstar_j:
                val_i = sorted_dist_indices[i, idx]
                val_j = sorted_dist_indices[j, idx2]
                if val_i < val_j:
                    idx += 1
                elif val_i > val_j:
                    idx2 += 1
                else:
                    count += 1
                    idx += 1
                    idx2 += 1
            common_neighs_mat[i,j] = count
            common_neighs_mat[j,i] = count
            common_neighs_array[ind_spar] = count
        else:
            common_neighs_mat[i,j] = common_neighs_mat[j,i]
            common_neighs_array[ind_spar] = common_neighs_mat[j,i]

    return common_neighs_array, common_neighs_mat

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