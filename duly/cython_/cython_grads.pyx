import cython
import numpy as np
import time
cimport numpy as np

DTYPE = np.int
floatTYPE = np.float
boolTYPE = np.bool

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t


@cython.boundscheck(False)
@cython.cdivision(True)
def compute_deltaFs_from_coords_and_grads(np.ndarray[floatTYPE_t, ndim = 2] X,
                                    np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                    np.ndarray[DTYPE_t, ndim = 1] kstar,
                                    np.ndarray[DTYPE_t, ndim = 2] grads,
                                    np.ndarray[DTYPE_t, ndim = 2] grads_covmat):
    # TODO: function should be checked! It should take the gradients and compute the deltaFs and the errors
    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs = np.zeros((N, kstar_max))
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs_var = np.zeros((N, kstar_max))
    cdef int i, j, k, dim, ki, dim2
    cdef int ind_j, ind_ki
    cdef floatTYPE_t Fij,Fij_sq, rk_sq

    for i in range(N):
        ki = kstar[i]-1
        ind_ki = dist_indices[i, ki]

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



@cython.boundscheck(False)
@cython.cdivision(True)
def compute_deltaFs_from_coords(np.ndarray[floatTYPE_t, ndim = 2] X,
                                np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                np.ndarray[DTYPE_t, ndim = 1] kstar,
                                floatTYPE_t id_selected):
    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs = np.zeros((N, kstar_max))
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs_var = np.zeros((N, kstar_max))
    cdef int i, j, k, dim, ki
    cdef int ind_j, ind_k, ind_ki
    cdef floatTYPE_t Fij,Fij_sq, Fijk, rk_sq, kifloat
    cdef floatTYPE_t dp2 = id_selected + 2.


    for i in range(N):
        ki = kstar[i]-1
        kifloat = float(ki)

        ind_ki = dist_indices[i, ki]

        rk_sq = 0.
        for dim in range(dims):
           rk_sq += (X[ind_ki, dim] - X[i, dim])**2
        # rk_sq = np.linalg.norm(X[ind_ki] - X[i])**2

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

@cython.boundscheck(False)
@cython.cdivision(True)
def compute_grads_and_var_from_coords(np.ndarray[floatTYPE_t, ndim = 2] X,
                                      np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                      np.ndarray[DTYPE_t, ndim = 1] kstar,
                                      floatTYPE_t id_selected):
    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads_var = np.zeros((N, dims))

    cdef int i, j, dim, ki, dim2
    cdef int ind_j, ind_ki
    cdef floatTYPE_t rk_sq, kifloat
    cdef floatTYPE_t dp2 = id_selected + 2.

    for i in range(N):
        ki = kstar[i] - 1

        kifloat = float(ki)

        ind_ki = dist_indices[i, ki]

        rk_sq = 0.
        for dim in range(dims):
            rk_sq += (X[ind_ki, dim] - X[i, dim]) ** 2

        # compute gradients and variance of gradients together
        for dim in range(dims):
            for j in range(ki):
                ind_j = dist_indices[i, j + 1]

                grads[i, dim] += (X[ind_j, dim] - X[i, dim])
                grads_var[i, dim] += (X[ind_j, dim] - X[i, dim]) * (X[ind_j, dim] - X[i, dim])

            grads[i, dim] = grads[i, dim] / kifloat * dp2 / rk_sq

            grads_var[i, dim] = grads_var[i, dim] / kifloat / kifloat * dp2 / rk_sq * dp2 / rk_sq \
                                - grads[i, dim] * grads[i, dim] / kifloat

    return grads, grads_var

@cython.boundscheck(False)
@cython.cdivision(True)
def compute_grads_and_covmat_from_coords(np.ndarray[floatTYPE_t, ndim = 2] X,
                                         np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                         np.ndarray[DTYPE_t, ndim = 1] kstar,
                                         floatTYPE_t id_selected):
    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 3] grads_covmat = np.zeros((N, dims, dims))

    cdef int i, j, dim, ki, dim2
    cdef int ind_j, ind_ki
    cdef floatTYPE_t rk_sq, kifloat
    cdef floatTYPE_t dp2 = id_selected + 2.

    for i in range(N):
        ki = kstar[i] - 1

        kifloat = float(ki)

        ind_ki = dist_indices[i, ki]

        rk_sq = 0.
        for dim in range(dims):
            rk_sq += (X[ind_ki, dim] - X[i, dim]) ** 2

        # compute gradients
        for dim in range(dims):
            for j in range(ki):
                ind_j = dist_indices[i, j + 1]

                grads[i, dim] += (X[ind_j, dim] - X[i, dim])

            grads[i, dim] = grads[i, dim] / kifloat * dp2 / rk_sq

        # compute covariance matrix of gradients
        for dim in range(dims):
            for dim2 in range(dims):
                for j in range(ki):
                    ind_j = dist_indices[i, j + 1]

                    grads_covmat[i, dim, dim2] += (X[ind_j, dim] - X[i, dim]) * (X[ind_j, dim2] - X[i, dim2])

                grads_covmat[i, dim, dim2] = grads_covmat[i, dim, dim2] / kifloat / kifloat * dp2 / rk_sq * dp2 / rk_sq \
                                             - grads[i, dim] * grads[i, dim2] / kifloat

    return grads, grads_covmat



