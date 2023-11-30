import cython
import time
cimport numpy as cnp
import numpy as np


from libc.math cimport (  #c FUNCTIONS FASTER THAN NUMPY
    fabs,
    nearbyint,
    exp,
    sqrt
)

from libc.stdio cimport printf


# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def provaprova( double[:,:,:] grads_covmat,
                double[:,:] neigh_vector_diffs,
                long[:,:] nind_list,          # nspar x 2
                double[:,:] p,                  # pearson_correlation matrix (NxN)
                double[:] Fij_var_array):
    #cdef DTYPE_t N = p.shape[0]
    #cdef DTYPE_t N = p.shape[0]

    cdef int dims = neigh_vector_diffs.shape[1]
    cdef int nspar = nind_list.shape[0]

    Gamma_nonview       = np.zeros((nspar,nspar), dtype=np.float_)   # corss covariance matrix
    inv_Gamma_nonview   = np.zeros(nspar, dtype=np.float_)       # inverse of diagonal of Gamma matrix
    cdef double[:,::1] Gamma = Gamma_nonview
    cdef double[::1] inv_Gamma = inv_Gamma_nonview
    
    #support
    denom_nonview   = np.zeros(nspar, dtype=np.float_)
    cdef double[:] denom = denom_nonview

    cdef double gamma, ptot, tmpi, tmpj, tmpl, tmpm, temp
    cdef int i,j,l,m,a,b,dim1,dim2    

    #for a in range(nspar):
    for a in range(10):
        print("a = ",a,flush=True)
        i = nind_list[a, 0]
        j = nind_list[a, 1]
        Gamma[a,a] = Fij_var_array[a]
        inv_Gamma[a] = Fij_var_array[a]
        denom[a] += Fij_var_array[a]*Fij_var_array[a]
        #for b in range(a+1, nspar):
        for b in range(a+1,10):
            print("b = ",b,flush=True)
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
                printf("tmpl\n")
                # if fabs(tmpi*tmpl) < 0:
                #     print("occazz")
                # if sqrt(fabs(tmpi*tmpl)) < 0:
                #     print("oooooooooccazz")
                #gamma += fabs(tmpi*tmpl)
                #gamma += sqrt(2.2)
                #gamma += np.sqrt(np.abs(tmpi*tmpl))

                #gamma += sqrt(fabs(tmpi*tmpl))
                #gamma += p[i,l]*sqrt(fabs(tmpi*tmpl))

        print("Faccrocazz 3",flush=True)    
                # 
                # if p[i,m] > 0:
                #     for dim1 in range(dims):
                #         for dim2 in range(dims):
                #             # r_ij @ varm @ r_lm
                #             tmpm += neigh_vector_diffs[a,dim1]*grads_covmat[m,dim1,dim2]*neigh_vector_diffs[b,dim2]
                #     gamma += p[i,m]*np.sqrt(tmpi*tmpm)            



    print("Faccrocazz 2",flush=True)

    return 2


@cython.boundscheck(False)
@cython.cdivision(True)
# def return_deltaFs_cross_covariance(double[:,:,::1] grads_covmat,
#                                     double[:,::1] neigh_vector_diffs,
#                                     long[:,::1] nind_list,          # nspar x 2
#                                     double[:,::1] p,                  # pearson_correlation matrix (NxN)
#                                     double[::1] Fij_var_array):
def return_deltaFs_cross_covariance(double[:,:,:] grads_covmat,
                                    double[:,:] neigh_vector_diffs,
                                    long[:,:] nind_list,          # nspar x 2
                                    double[:,:] p,                  # pearson_correlation matrix (NxN)
                                    double[:] Fij_var_array):
    #cdef DTYPE_t N = p.shape[0]
    print("Faccrocazz")

    cdef int dims = neigh_vector_diffs.shape[1]
    cdef int nspar = nind_list.shape[0]
    Gamma_nonview       = np.zeros((nspar,nspar), dtype=np.float_)   # corss covariance matrix
    inv_Gamma_nonview   = np.zeros(nspar, dtype=np.float_)       # inverse of diagonal of Gamma matrix
    #cdef double[:,::1] Gamma = Gamma_nonview
    #cdef double[::1] inv_Gamma = inv_Gamma_nonview
    cdef double[:,:] Gamma = Gamma_nonview
    cdef double[:] inv_Gamma = inv_Gamma_nonview

    #support
    vari_nonview    = np.zeros((dims,dims), dtype=np.float_)
    varj_nonview    = np.zeros((dims,dims), dtype=np.float_)
    varl_nonview    = np.zeros((dims,dims), dtype=np.float_)
    varm_nonview    = np.zeros((dims,dims), dtype=np.float_)
    r_ij_nonview    = np.zeros(dims, dtype=np.float_)
    r_lm_nonview    = np.zeros(dims, dtype=np.float_)
    denom_nonview   = np.zeros(nspar, dtype=np.float_)
    # cdef double[:,::1] vari = vari_nonview
    # cdef double[:,::1] varj = varj_nonview
    # cdef double[:,::1] varl = varl_nonview
    # cdef double[:,::1] varm = varm_nonview
    # cdef double[::1] r_ij = r_ij_nonview
    # cdef double[::1] r_lm = r_lm_nonview
    # cdef double[::1] denom = denom_nonview

    cdef double[:,:] vari = vari_nonview
    cdef double[:,:] varj = varj_nonview
    cdef double[:,:] varl = varl_nonview
    cdef double[:,:] varm = varm_nonview
    cdef double[:] r_ij = r_ij_nonview
    cdef double[:] r_lm = r_lm_nonview
    cdef double[:] denom = denom_nonview
    
    cdef double gamma, ptot, tmpi, tmpj, tmpl, tmpm, temp
    cdef int a,b,i,j,l,m,dim1,dim2


    for a in range(nspar):
        print("a = ",a)
        i = nind_list[a, 0]
        j = nind_list[a, 1]
        Gamma[a,a] = Fij_var_array[a]
        inv_Gamma[a] = Fij_var_array[a]
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
                r_ij = neigh_vector_diffs[a]
                r_lm = neigh_vector_diffs[b]
                vari = grads_covmat[i]
                varl = grads_covmat[l]
                for dim1 in range(dims):
                    for dim2 in range(dims):
                        tmpi += r_ij[dim1]*vari[dim1,dim2]*r_lm[dim2]
                for dim1 in range(dims):
                    for dim2 in range(dims):
                        tmpl += r_ij[dim1]*varl[dim1,dim2]*r_lm[dim2]
                gamma += p[i,l]*np.sqrt(tmpi*tmpl)
                if p[i,m] > 0:
                    varm = grads_covmat[m]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpm += r_ij[dim1]*varm[dim1,dim2]*r_lm[dim2]
                    gamma += p[i,m]*np.sqrt(tmpi*tmpm)





            else:
                if p[i,m] > 0:
                    ptot += p[i,m]
                    r_ij = neigh_vector_diffs[a]
                    r_lm = neigh_vector_diffs[b]
                    vari = grads_covmat[i]
                    varm = grads_covmat[m]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpi += r_ij[dim1]*vari[dim1,dim2]*r_lm[dim2]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpm += r_ij[dim1]*varm[dim1,dim2]*r_lm[dim2]
                    gamma += p[i,m]*np.sqrt(tmpi*tmpm)
            
            if p[j,l] > 0:
                varj = grads_covmat[j]
                if ptot == 0:
                    r_ij = neigh_vector_diffs[a]
                    r_lm = neigh_vector_diffs[b]
                    ptot += p[j,l]
                    varl = grads_covmat[l]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpj += r_ij[dim1]*varj[dim1,dim2]*r_lm[dim2]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpl += r_ij[dim1]*varl[dim1,dim2]*r_lm[dim2]
                    gamma += p[j,l]*np.sqrt(tmpj*tmpl)
                    if p[j,m] > 0:
                        varm = grads_covmat[m]
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                tmpm += r_ij[dim1]*varm[dim1,dim2]*r_lm[dim2]
                        gamma += p[j,m]*np.sqrt(tmpj*tmpm)
                else:
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpj += r_ij[dim1]*varj[dim1,dim2]*r_lm[dim2]
                    if tmpl == 0:
                        varl = grads_covmat[l]
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                tmpl += r_ij[dim1]*varl[dim1,dim2]*r_lm[dim2]
                    gamma += p[j,l]*np.sqrt(tmpj*tmpl)
                    if p[j,m] > 0:
                        if tmpm == 0:
                            varm = grads_covmat[m]
                            for dim1 in range(dims):
                                for dim2 in range(dims):
                                    tmpm += r_ij[dim1]*varm[dim1,dim2]*r_lm[dim2]
                        gamma += p[j,m]*np.sqrt(tmpj*tmpm)
            else:
                if p[j,m] > 0:
                    varj = grads_covmat[j]
                    if ptot == 0:
                        r_ij = neigh_vector_diffs[a]
                        r_lm = neigh_vector_diffs[b]
                        ptot += p[j,m]
                    if tmpm == 0:
                        varm = grads_covmat[m]
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                tmpm += r_ij[dim1]*varm[dim1,dim2]*r_lm[dim2]                    
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpj += r_ij[dim1]*varj[dim1,dim2]*r_lm[dim2]
                    gamma += p[j,m]*np.sqrt(tmpj*tmpm)
            if ptot != 0:
                Gamma[a,b] = gamma
                Gamma[b,a] = gamma
                denom[a] += gamma*gamma
                denom[b] += gamma*gamma
        
    for a in range(nspar):
        inv_Gamma[a] /= denom[a]

    return Gamma, inv_Gamma


#     get_vector_diffs = sparse.lil_matrix((N, N),dtype=DTYPE)
#     mask = sparse.lil_matrix((N, N),dtype=np.bool_)


# ----------------------------------------------------------------------------------------------


