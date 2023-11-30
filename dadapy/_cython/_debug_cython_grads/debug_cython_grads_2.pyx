import cython
import numpy as np
import time
cimport numpy as np

from libc.math cimport (  # absolute values for floats, needed when using PBC
    fabs,
    nearbyint,
    exp,
    sqrt
)

DTYPE = np.int
floatTYPE = np.float
boolTYPE = np.bool

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def provaprova( double[:,:,::1] grads_covmat,
                double[:,::1] neigh_vector_diffs,
                long[:,::1] nind_list,          # nspar x 2
                double[:,::1] p,                  # pearson_correlation matrix (NxN)
                double[::1] Fij_var_array):
    #cdef DTYPE_t N = p.shape[0]
    #cdef DTYPE_t N = p.shape[0]

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


@cython.boundscheck(False)
@cython.cdivision(True)
def return_deltaFs_cross_covariance(np.ndarray[floatTYPE_t, ndim = 3] grads_covmat,
                                    np.ndarray[floatTYPE_t, ndim = 2] neigh_vector_diffs,
                                    np.ndarray[DTYPE_t, ndim = 2] nind_list,        # nspar x 2
                                    np.ndarray[floatTYPE_t, ndim = 2] p,            # pearson_correlation matrix (NxN)
                                    np.ndarray[floatTYPE_t, ndim = 1] Fij_var_array):
    #cdef DTYPE_t N = p.shape[0]
    cdef DTYPE_t dims = neigh_vector_diffs.shape[1]
    cdef DTYPE_t nspar = nind_list.shape[0]
    cdef double[:,:] Gamma = np.zeros((nspar,nspar), dtype=np.float_)   # corss covariance matrix
    cdef double[:] inv_Gamma = np.zeros(nspar, dtype=np.float_)         # inverse of diagonal of Gamma matrix

    #support
    cdef np.ndarray[floatTYPE_t, ndim=2] vari = np.zeros((dims,dims), dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim=2] varj = np.zeros((dims,dims), dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim=2] varl = np.zeros((dims,dims), dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim=2] varm = np.zeros((dims,dims), dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim=1] r_ij = np.zeros(dims, dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim=1] r_lm = np.zeros(dims, dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim=1] denom = np.zeros(nspar, dtype=np.float_)
    cdef floatTYPE_t gamma, ptot, tmpi, tmpj, tmpl, tmpm, temp
    cdef DTYPE_t a,b,i,j,l,m,dim1,dim2


    for a in range(nspar):
        #print("a=", a, flush=True)
        if a%100 == 0:
            print("a=", a, flush=True)
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
                
                gamma += p[i,l]*sqrt(fabs(tmpi*tmpl))
                if p[i,m] > 0:
                    varm = grads_covmat[m]
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpm += r_ij[dim1]*varm[dim1,dim2]*r_lm[dim2]
                    gamma += p[i,m]*sqrt(fabs(tmpi*tmpm))
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
                    gamma += p[i,m]*sqrt(fabs(tmpi*tmpm))
            
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
                    gamma += p[j,l]*sqrt(fabs(tmpj*tmpl))
                    if p[j,m] > 0:
                        varm = grads_covmat[m]
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                tmpm += r_ij[dim1]*varm[dim1,dim2]*r_lm[dim2]
                        gamma += p[j,m]*sqrt(fabs(tmpj*tmpm))
                else:
                    for dim1 in range(dims):
                        for dim2 in range(dims):
                            tmpj += r_ij[dim1]*varj[dim1,dim2]*r_lm[dim2]
                    if tmpl == 0:
                        varl = grads_covmat[l]
                        for dim1 in range(dims):
                            for dim2 in range(dims):
                                tmpl += r_ij[dim1]*varl[dim1,dim2]*r_lm[dim2]
                    gamma += p[j,l]*sqrt(fabs(tmpj*tmpl))
                    if p[j,m] > 0:
                        if tmpm == 0:
                            varm = grads_covmat[m]
                            for dim1 in range(dims):
                                for dim2 in range(dims):
                                    tmpm += r_ij[dim1]*varm[dim1,dim2]*r_lm[dim2]
                        gamma += p[j,m]*sqrt(fabs(tmpj*tmpm))
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
                    gamma += p[j,m]*sqrt(fabs(tmpj*tmpm))
            if ptot != 0:
                Gamma[a, b] = gamma / 4.
                Gamma[b, a] = gamma / 4.
                denom[a] += gamma * gamma / 16.
                denom[b] += gamma * gamma / 16.
        
    for a in range(nspar):
        inv_Gamma[a] /= denom[a]

    return inv_Gamma