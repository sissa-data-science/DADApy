import cython
import numpy as np
import time
cimport numpy as np

DTYPE = np.int_
floatTYPE = np.float_
boolTYPE = np.bool_

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t #AAAAAAAAAAAAAAA FORSE OBSOLETI

from libc.math cimport (  #c FUNCTIONS FASTER THAN NUMPY
    fabs,
    nearbyint,
    exp,
    sqrt
)


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
    #    cdef np.ndarray[DTYPE_t, ndim = 2] nind_mat = np.zeros((N, N),dtype=DTYPE)

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
    cdef np.ndarray[floatTYPE_t, ndim = 1] distarray = np.ndarray((nspar,),dtype=floatTYPE)

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
    cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims))

    cdef DTYPE_t i, j, ind_spar, dim

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
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims))

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
    cdef DTYPE_t N = kstar.shape[0]
    cdef DTYPE_t nspar = nind_list.shape[0]

    cdef DTYPE_t i, j, ind_spar

    cdef np.ndarray[DTYPE_t, ndim=1] common_neighs_array = np.zeros(nspar, dtype=np.int_)
    cdef np.ndarray[DTYPE_t, ndim=2] common_neighs_mat = np.zeros((N,N), dtype=np.int_)

    #for i in range(N):
    #    ki = kstar[i]-1
    #    for k in range(ki):

    for ind_spar in range(nspar):
        i = nind_list[ind_spar, 0]
        j = nind_list[ind_spar, 1]
        if common_neighs_mat[j,i] == 0:
            common_neighs_mat[i,j] = np.in1d(dist_indices[i, :kstar[i]], dist_indices[j, :kstar[j]],
                                          assume_unique=True).sum()
            common_neighs_mat[j,i] = common_neighs_mat[i,j]
            common_neighs_array[ind_spar] = common_neighs_mat[i,j]
        else:
            common_neighs_mat[i,j] = common_neighs_mat[j,i]
            common_neighs_array[ind_spar] = common_neighs_mat[j,i]

    #        j = dist_indices[i, k+1]
    #        if common_neighs[i,j] == 0:
    #            common_neighs[i,j] = np.in1d(dist_indices[i,:ki+1],dist_indices[j,:kstar[j]],assume_unique=True).sum()
    #            common_neighs[j,i] = common_neighs[i,j]

    return common_neighs_array, common_neighs_mat

# ----------------------------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.cdivision(True)
def return_deltaFs_inv_cross_covariance(double[:,:,:] grads_covmat,
                                        double[:,:] neigh_vector_diffs,
                                        long[:,:] nind_list,          # nspar x 2
                                        double[:,:] p,                  # pearson_correlation matrix (NxN)
                                        double[:] Fij_var_array):
    cdef int dims = neigh_vector_diffs.shape[1]
    cdef int nspar = nind_list.shape[0]

    #Gamma_nonview       = np.zeros((nspar,nspar), dtype=np.float_)   # corss covariance matrix
    inv_Gamma_nonview   = np.zeros(nspar, dtype=np.float_)       # inverse of diagonal of Gamma matrix
    #cdef double[:,::1] Gamma = Gamma_nonview
    cdef double[::1] inv_Gamma = inv_Gamma_nonview
    
    #support
    denom_nonview   = np.zeros(nspar, dtype=np.float_)
    cdef double[::1] denom = denom_nonview

    cdef double gamma, ptot, tmpi, tmpj, tmpl, tmpm, temp
    cdef int i,j,l,m,a,b,dim1,dim2    

    for a in range(nspar):
    #for a in range(10):
        #if a%100 == 0:
        #    print("a=", a, flush=True)
        #print("a = ",a,flush=True)
        i = nind_list[a, 0]
        j = nind_list[a, 1]
        #Gamma[a,a] = Fij_var_array[a]
        inv_Gamma[a] = Fij_var_array[a]
        denom[a] += Fij_var_array[a]*Fij_var_array[a]
        for b in range(a+1, nspar):
        #for b in range(a+1,10):
            #print("b = ",b,flush=True)
            l = nind_list[b, 0]
            m = nind_list[b, 1]
            gamma = 0
            ptot = 0
            tmpi = 0
            tmpj = 0
            tmpl = 0
            tmpm = 0
            if p[i,l] > 0:
                ptot += p[i,l]

                #for dim1 in range(dims):
                    #dim1*=1
                    #print("c",flush=True)
                    #print(neigh_vector_diffs[a,ii],flush=True)
                #for ii in range(dims):
                #     r_lm[ii] = neigh_vector_diffs[b,ii]
                # for ii in range(dims):
                #     for jj in range(dims):
                #         vari[ii,jj] = grads_covmat[i,ii,jj]
                # for ii in range(dims):
                #     for jj in range(dims):
                #         varl[ii,jj] = grads_covmat[l,ii,jj]  

                for dim1 in range(dims):
                    for dim2 in range(dims):
                        # r_ij @ vari @ r_lm
                        tmpi += neigh_vector_diffs[a,dim1]*grads_covmat[i,dim1,dim2]*neigh_vector_diffs[b,dim2]        
                #print("tmpi = ",tmpi,flush=True)
                for dim1 in range(dims):
                    for dim2 in range(dims):
                        # r_ij @ varl @ r_lm
                        tmpl += neigh_vector_diffs[a,dim1]*grads_covmat[l,dim1,dim2]*neigh_vector_diffs[b,dim2]
                gamma += p[i,l]*sqrt(fabs(tmpi*tmpl))
                #printf("tmpl\n")
                # if fabs(tmpi*tmpl) < 0:
                #     print("occazz")
                # if sqrt(fabs(tmpi*tmpl)) < 0:
                #     print("oooooooooccazz")
                #gamma += fabs(tmpi*tmpl)
                #gamma += sqrt(2.2)
                #gamma += sqrt(fabs(tmpi*tmpl))

                #gamma += sqrt(fabs(tmpi*tmpl))
                if p[i,m] > 0:
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            # r_ij @ varm @ r_lm
                            tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    gamma += p[i,m]*sqrt(fabs(tmpi*tmpm))
            else:
                if p[i,m] > 0:
                    ptot += p[i,m]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            # r_ij @ vari @ r_lm
                            tmpi += neigh_vector_diffs[a,dim1]*grads_covmat[i,dim1,dim2]*neigh_vector_diffs[b,dim2]        
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            # r_ij @ varm @ r_lm
                            tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]        
                    gamma += p[i,m]*sqrt(fabs(tmpi*tmpm))

            if p[j,l] > 0:
                if ptot == 0:
                    ptot += p[j,l]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            # r_ij @ varj @ r_lm
                            tmpj += neigh_vector_diffs[a,dim1]*grads_covmat[j,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            # r_ij @ varl @ r_lm
                            tmpl += neigh_vector_diffs[a,dim1]*grads_covmat[l,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    gamma += p[j,l]*sqrt(fabs(tmpj*tmpl))
                    if p[j,m] > 0:
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                # r_ij @ varm @ r_lm
                                tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]
                        gamma += p[j,m]*sqrt(fabs(tmpj*tmpm))
                else:
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            # r_ij @ varj @ r_lm
                            tmpj += neigh_vector_diffs[a,dim1]*grads_covmat[j,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    if tmpl == 0:
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                # r_ij @ varl @ r_lm
                                tmpl += neigh_vector_diffs[a,dim1]*grads_covmat[l,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    gamma += p[j,l]*sqrt(fabs(tmpj*tmpl))
                    if p[j,m] > 0:
                        if tmpm == 0:
                            for dim1 in range(dims):
                                for dim2 in range(dims):
                                    # r_ij @ varm @ r_lm
                                    tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]
                        gamma += p[j,m]*sqrt(fabs(tmpj*tmpm))
            else:
                if p[j,m] > 0:
                    if ptot == 0:
                        ptot += p[j,m]
                    if tmpm == 0:
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                # r_ij @ varm @ r_lm
                                tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            # r_ij @ varj @ r_lm
                            tmpj += neigh_vector_diffs[a,dim1]*grads_covmat[j,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    gamma += p[j,m]*sqrt(fabs(tmpj*tmpm))
            if ptot != 0:
                #Gamma[a, b] = gamma / 4.
                #Gamma[b, a] = gamma / 4.
                # AAAAAAAAAAAAAAA AGGIUNGERE UN'OPZIONE CHE DECIDE SE OUTPUT GAMMA O SOLO INVGAMMA
                denom[a] += gamma * gamma / 16.
                denom[b] += gamma * gamma / 16.
        
    for a in range(nspar):
        inv_Gamma[a] /= denom[a]

    #return Gamma, inv_Gamma
    return np.asarray(inv_Gamma)


# ----------------------------------------------------------------------------------------------



@cython.boundscheck(False)
@cython.cdivision(True)
def return_deltaFs_cross_covariance_and_inv(double[:,:,:] grads_covmat,
                                            double[:,:] neigh_vector_diffs,
                                            long[:,:] nind_list,          # nspar x 2
                                            double[:,:] p,                  # pearson_correlation matrix (NxN)
                                            double[:] Fij_var_array):
    cdef int dims = neigh_vector_diffs.shape[1]
    cdef int nspar = nind_list.shape[0]

    Gamma_nonview       = np.zeros((nspar,nspar), dtype=np.float_)   # corss covariance matrix
    inv_Gamma_nonview   = np.zeros(nspar, dtype=np.float_)       # inverse of diagonal of Gamma matrix
    
    cdef double[:,::1] Gamma = Gamma_nonview
    cdef double[::1] inv_Gamma = inv_Gamma_nonview
    
    # support
    denom_nonview  = np.zeros(nspar, dtype=np.float_)
    cdef double[::1] denom = denom_nonview

    cdef double gamma, ptot, tmpi, tmpj, tmpl, tmpm, temp
    cdef int i,j,l,m,a,b,dim1,dim2    

    for a in range(nspar):
        i = nind_list[a, 0]
        j = nind_list[a, 1]
        inv_Gamma[a] = Fij_var_array[a]
        Gamma[a, a] = Fij_var_array[a]
        denom[a] += Fij_var_array[a]*Fij_var_array[a]
        for b in range(a+1, nspar):
            l = nind_list[b, 0]
            m = nind_list[b, 1]
            gamma = 0
            ptot = 0
            tmpi = 0
            tmpj = 0
            tmpl = 0
            tmpm = 0
            if p[i,l] > 0:
                ptot += p[i,l]
                for dim1 in range(dims):
                    for dim2 in range(dims):
                        tmpi += neigh_vector_diffs[a,dim1]*grads_covmat[i,dim1,dim2]*neigh_vector_diffs[b,dim2]        
                for dim1 in range(dims):
                    for dim2 in range(dims):
                        tmpl += neigh_vector_diffs[a,dim1]*grads_covmat[l,dim1,dim2]*neigh_vector_diffs[b,dim2]
                gamma += p[i,l]*sqrt(fabs(tmpi*tmpl))
                if p[i,m] > 0:
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    gamma += p[i,m]*sqrt(fabs(tmpi*tmpm))
            else:
                if p[i,m] > 0:
                    ptot += p[i,m]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpi += neigh_vector_diffs[a,dim1]*grads_covmat[i,dim1,dim2]*neigh_vector_diffs[b,dim2]        
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]        
                    gamma += p[i,m]*sqrt(fabs(tmpi*tmpm))

            if p[j,l] > 0:
                if ptot == 0:
                    ptot += p[j,l]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpj += neigh_vector_diffs[a,dim1]*grads_covmat[j,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpl += neigh_vector_diffs[a,dim1]*grads_covmat[l,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    gamma += p[j,l]*sqrt(fabs(tmpj*tmpl))
                    if p[j,m] > 0:
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]
                        gamma += p[j,m]*sqrt(fabs(tmpj*tmpm))
                else:
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpj += neigh_vector_diffs[a,dim1]*grads_covmat[j,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    if tmpl == 0:
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                tmpl += neigh_vector_diffs[a,dim1]*grads_covmat[l,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    gamma += p[j,l]*sqrt(fabs(tmpj*tmpl))
                    if p[j,m] > 0:
                        if tmpm == 0:
                            for dim1 in range(dims):
                                for dim2 in range(dims):
                                    tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]
                        gamma += p[j,m]*sqrt(fabs(tmpj*tmpm))
            else:
                if p[j,m] > 0:
                    if ptot == 0:
                        ptot += p[j,m]
                    if tmpm == 0:
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpj += neigh_vector_diffs[a,dim1]*grads_covmat[j,dim1,dim2]*neigh_vector_diffs[b,dim2]
                    gamma += p[j,m]*sqrt(fabs(tmpj*tmpm))

            if ptot != 0:
                Gamma[a, b] = gamma / 4.
                Gamma[b, a] = gamma / 4.

                denom[a] += gamma * gamma / 16.
                denom[b] += gamma * gamma / 16.
        
    for a in range(nspar):
        inv_Gamma[a] /= denom[a]

    return np.asarray(Gamma), np.asarray(inv_Gamma)
    #return np.asarray(inv_Gamma)


# ----------------------------------------------------------------------------------------------






# @cython.boundscheck(False)
# @cython.cdivision(True)
# def return_deltaFs_gradient_semisum(np.ndarray[floatTYPE_t, ndim = 2] vector_diffs,
#                                     get_vector_diffs,
#                                     np.ndarray[DTYPE_t, ndim = 2] dist_indices,
#                                     np.ndarray[DTYPE_t, ndim = 1] kstar):
#     cdef DTYPE_t N = X.shape[0]
#     cdef DTYPE_t dims = X.shape[1]
#     cdef DTYPE_t kstar_max = np.max(kstar)
#     cdef DTYPE_t nspar = kstar.sum()-N
#     cdef np.ndarray[floatTYPE_t, ndim = 2] vector_diffs = np.ndarray((nspar, dims))

#     get_vector_diffs = sparse.lil_matrix((N, N),dtype=DTYPE)
#     mask = sparse.lil_matrix((N, N),dtype=np.bool_)

#     cdef DTYPE_t i, j, k, dim, ki
#     cdef DTYPE_t ind_spar

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
    # NOT USED AT THE MOMENT!
    # TODO: function should be checked! It should take the gradients and compute the deltaFs and the errors
    cdef DTYPE_t N = X.shape[0]
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs = np.zeros((N, kstar_max))
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs_var = np.zeros((N, kstar_max))
    cdef DTYPE_t i, j, k, dim, ki, dim2
    cdef DTYPE_t ind_j
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
# NOT USED AT THE MOMENT!
    cdef DTYPE_t N = X.shape[0]
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs = np.zeros((N, kstar_max))
    cdef np.ndarray[floatTYPE_t, ndim = 2] delta_Fijs_var = np.zeros((N, kstar_max))
    cdef DTYPE_t i, j, k, dim, ki
    cdef DTYPE_t ind_j, ind_k
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

    cdef DTYPE_t N = X.shape[0]
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads_var = np.zeros((N, dims))
    
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

    cdef DTYPE_t N = X.shape[0]
    cdef DTYPE_t dims = X.shape[1]
    cdef DTYPE_t kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 3] grads_covmat = np.zeros((N, dims, dims))

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
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads_var = np.zeros((N, dims))
    
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
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef np.ndarray[floatTYPE_t, ndim = 3] grads_covmat = np.zeros((N, dims, dims))

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