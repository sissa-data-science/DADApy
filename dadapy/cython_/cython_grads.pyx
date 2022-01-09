import time

import cython
import numpy as np

cimport numpy as np

DTYPE = np.int
floatTYPE = np.float
boolTYPE = np.bool

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_ind(np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                     np.ndarray[DTYPE_t, ndim = 1] kstar):
    cdef int N = kstar.shape[0]
    cdef int kstar_max = np.max(kstar)
    cdef int nspar = kstar.sum() - N
    cdef np.ndarray[DTYPE_t, ndim = 2] nind_list = np.ndarray((nspar, 2), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] nind_iptr = np.ndarray(shape=(N + 1,), dtype=DTYPE)
    #    cdef np.ndarray[DTYPE_t, ndim = 2] nind_mat = np.zeros((N, N),dtype=DTYPE)

    cdef int i, j, k, ind_spar, ki

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
    cdef int N = len(kstar)
    cdef int nspar = kstar.sum() - N
    cdef np.ndarray[floatTYPE_t, ndim = 1] distarray = np.ndarray((nspar,),dtype=floatTYPE)

    cdef int i, j, ind_spar

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
    cdef int dims = X.shape[1]
    cdef int nspar = nind_list.shape[0]
    cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims))

    cdef int i, j, ind_spar, dim

    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]
        for dim in range(dims):
            vector_diffs[ind_spar, dim] = X[j, dim] - X[i, dim]

    # ind_spar = 0
    # for i in range(N):
    #     ki = kstar[i]-1
    #     for k in range(ki):
    #         j = dist_indices[i, k+1]
    #         if nind_mat[j,i] is not None:
    #             for dim in range(dims):
    #                 vector_diffs[ind_spar,dim] = -vector_diffs[nind_mat[j,i],dim]
    #         else:
    #             for dim in range(dims):
    #                 vector_diffs[ind_spar,dim] = X[j, dim] - X[i, dim]
    #         nind_mat[i,j] = ind_spar
    #         nind_list[ind_spar,0]=i
    #         nind_list[ind_spar,1]=j
    #         ind_spar += 1

    return vector_diffs

# ----------------------------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.cdivision(True)
def return_neigh_vector_diffs_periodic(np.ndarray[floatTYPE_t, ndim = 2] X,
                              np.ndarray[DTYPE_t, ndim = 2] nind_list,
                              np.ndarray[floatTYPE_t, ndim = 1] period):
    cdef int dims = X.shape[1]
    cdef int nspar = nind_list.shape[0]
    cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims))

    cdef int i, j, ind_spar, dim
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


    # ind_spar = 0
    # for i in range(N):
    #     ki = kstar[i]-1
    #     for k in range(ki):
    #         j = dist_indices[i, k+1]
    #         if nind_mat[j,i] is not None:
    #             for dim in range(dims):
    #                 vector_diffs[ind_spar,dim] = -vector_diffs[nind_mat[j,i],dim]
    #         else:
    #             for dim in range(dims):
    #                 vector_diffs[ind_spar,dim] = X[j, dim] - X[i, dim]
    #         nind_mat[i,j] = ind_spar
    #         nind_list[ind_spar,0]=i
    #         nind_list[ind_spar,1]=j
    #         ind_spar += 1

    return vector_diffs

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_common_neighs(np.ndarray[DTYPE_t, ndim = 1] kstar,
                         np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                         np.ndarray[DTYPE_t, ndim = 2] nind_list):
    cdef int N = kstar.shape[0]
    cdef int nspar = nind_list.shape[0]

    cdef int i, j, ind_spar

    cdef np.ndarray[DTYPE_t, ndim=1] common_neighs = np.zeros(nspar, dtype=np.int_)

    #for i in range(N):
    #    ki = kstar[i]-1
    #    for k in range(ki):

    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]
        common_neighs[ind_spar] = np.in1d(dist_indices[i, :kstar[i]], dist_indices[j, :kstar[j]],
                                          assume_unique=True).sum()

    #        j = dist_indices[i, k+1]
    #        if common_neighs[i,j] == 0:
    #            common_neighs[i,j] = np.in1d(dist_indices[i,:ki+1],dist_indices[j,:kstar[j]],assume_unique=True).sum()
    #            common_neighs[j,i] = common_neighs[i,j]

    return common_neighs

# ----------------------------------------------------------------------------------------------

# @cython.boundscheck(False)
# @cython.cdivision(True)
# def return_deltaFs_gradient_semisum(np.ndarray[floatTYPE_t, ndim = 2] vector_diffs,
#                                     get_vector_diffs,
#                                     np.ndarray[DTYPE_t, ndim = 2] dist_indices,
#                                     np.ndarray[DTYPE_t, ndim = 1] kstar):
#     cdef int N = X.shape[0]
#     cdef int dims = X.shape[1]
#     cdef int kstar_max = np.max(kstar)
#     cdef int nspar = kstar.sum()-N
#     cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims))

#     get_vector_diffs = sparse.lil_matrix((N, N),dtype=DTYPE)
#     mask = sparse.lil_matrix((N, N),dtype=np.bool_)

#     cdef int i, j, k, dim, ki
#     cdef int ind_spar

#     ind_spar = 0
#     for i in range(N):
#         ki = kstar[i]-1
#         for k in range(ki):
#             j = dist_indices[i, k+1]
#             if mask[j,i] == True:
#                 for dim in range(dims):
#                     vector_diffs[ind_spar,dim] = -vector_diffs[get_vector_diffs[j,i],dim]
#             else:
#                 for dim in range(dims):
#                     vector_diffs[ind_spar,dim] = X[j, dim] - X[i, dim]
#             mask[i,j] = True
#             get_vector_diffs[i,j] = ind_spar
#             ind_spar += 1
#     assert (ind_spar == nspar)

#     return vector_diffs, get_vector_diffs


@cython.boundscheck(False)
@cython.cdivision(True)
def return_deltaFs_from_coords_and_grads(np.ndarray[floatTYPE_t, ndim = 2] X,
                                    np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                    np.ndarray[DTYPE_t, ndim = 1] kstar,
                                    np.ndarray[DTYPE_t, ndim = 2] grads,
                                    np.ndarray[DTYPE_t, ndim = 3] grads_covmat):
    # TODO: function should be checked! It should take the gradients and compute the deltaFs and the errors
    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs = np.zeros((N, kstar_max))
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs_var = np.zeros((N, kstar_max))
    cdef int i, j, k, dim, ki, dim2
    cdef int ind_j
    cdef floatTYPE_t Fij,Fij_sq, rk_sq

    for i in range(N):
        ki = kstar[i]-1

        for j in range(ki):
            ind_j = dist_indices[i, j+1]

            # deltaFij and its estimated variance
            Fij = 0.
            Fij_sq = 0.

            # simple contraction of gradient with deltaXij
            for dim in range(dims):
                Fij += grads[i, dim] * (X[ind_j, dim] - X[i, dim])

            # contraction deltaXij * covariance * deltaXij
            for dim in range(dims):
                for dim2 in range(dims):
                    Fij_sq += (X[ind_j, dim] - X[i, dim])*grads_covmat[i, dim, dim2] *(X[ind_j, dim2] - X[i, dim2])

            delta_Fijs[i, j] = Fij
            delta_Fijs_var[i, j] = Fij_sq

    delta_Fijs_list = [delta_Fijs[i, :kstar[i]-1] for i in range(N)]
    delta_Fijs_var_list = [delta_Fijs_var[i, :kstar[i]-1] for i in range(N)]

    return delta_Fijs_list, delta_Fijs_var_list

# ----------------------------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.cdivision(True)
def return_deltaFs_from_coords(np.ndarray[floatTYPE_t, ndim = 2] X,
                                np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                np.ndarray[DTYPE_t, ndim = 1] kstar,
                                floatTYPE_t id_selected):
    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs = np.zeros((N, kstar_max))
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs_var = np.zeros((N, kstar_max))
    cdef int i, j, k, dim, ki
    cdef int ind_j, ind_k
    cdef floatTYPE_t Fij,Fij_sq, Fijk, rk_sq, kifloat
    cdef floatTYPE_t dp2 = id_selected + 2.


    for i in range(N):
        ki = kstar[i]-1
        kifloat = float(ki)

        rk_sq = 0.
        for dim in range(dims):
           rk_sq += (X[dist_indices[i, ki+1], dim] - X[i, dim])**2

        for j in range(ki):
            ind_j = dist_indices[i, j+1]

            Fij = 0.
            Fij_sq = 0.

            for k in range(ki):
                ind_k = dist_indices[i, k+1]

                Fijk = 0.
                # computing the dot product this way is necessary to avoid any
                # call to python or numpy
                for dim in range(dims):
                    Fijk += (X[ind_j, dim] - X[i, dim])*(X[ind_k, dim] - X[i, dim])

                Fij += Fijk
                Fij_sq += Fijk**2

            Fij = Fij / kifloat * dp2/rk_sq

            Fij_sq = Fij_sq / kifloat / kifloat * (dp2/rk_sq) * (dp2/rk_sq) - Fij**2 / kifloat

            delta_Fijs[i, j] = Fij
            delta_Fijs_var[i, j] = Fij_sq

    delta_Fijs_list = [delta_Fijs[i, :kstar[i]-1] for i in range(N)]
    delta_Fijs_var_list = [delta_Fijs_var[i, :kstar[i]-1] for i in range(N)]

    return delta_Fijs_list, delta_Fijs_var_list

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def return_grads_and_var_from_coords(  np.ndarray[floatTYPE_t, ndim = 2] X,
                                        np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                        np.ndarray[DTYPE_t, ndim = 1] kstar,
                                        floatTYPE_t id_selected):

    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads_var = np.zeros((N, dims))
    
    cdef int i, j, dim, ki, dim2
    cdef int ind_j
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

    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 3] grads_covmat = np.zeros((N, dims, dims))

    cdef int i, j, dim, ki, dim2
    cdef int ind_j
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

    cdef int N = kstar.shape[0]
    cdef int dims = neigh_vector_diffs.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads_var = np.zeros((N, dims))
    
    cdef int i, j, dim, ki, dim2
    cdef int ind_j
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

    cdef int N = kstar.shape[0]
    cdef int dims = neigh_vector_diffs.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 3] grads_covmat = np.zeros((N, dims, dims))

    cdef int i, j, dim, ki, dim2
    cdef int ind_j
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