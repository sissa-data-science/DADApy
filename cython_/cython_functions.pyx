import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp

DTYPE = np.int
floatTYPE = np.float

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

@cython.boundscheck(False)
@cython.cdivision(True)
def compute_deltaFs_from_coords(np.ndarray[floatTYPE_t, ndim = 2] X,
                                np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                np.ndarray[DTYPE_t, ndim = 1] kstar, floatTYPE_t id_selected):
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
        ki = kstar[i]
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

            Fij_sq = Fij_sq / kifloat * (dp2/rk_sq)**2 - Fij**2

            delta_Fijs[i, j] = Fij
            delta_Fijs_var[i, j] = Fij_sq

    delta_Fijs_list = [delta_Fijs[i, :kstar[i]] for i in range(N)]
    delta_Fijs_var_list = [delta_Fijs_var[i, :kstar[i]] for i in range(N)]

    return delta_Fijs_list, delta_Fijs_var_list


@cython.boundscheck(False)
@cython.cdivision(True)
def compute_grads_from_coords(np.ndarray[floatTYPE_t, ndim = 2] X,
                                np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                                np.ndarray[DTYPE_t, ndim = 1] kstar, floatTYPE_t id_selected):

    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int kstar_max = np.max(kstar)
    cdef np.ndarray[floatTYPE_t, ndim = 2] grads = np.zeros((N, dims))
    cdef int i, j, dim, ki
    cdef int ind_j, ind_ki
    cdef floatTYPE_t rk_sq, kifloat
    cdef floatTYPE_t dp2 = id_selected + 2.

    for i in range(N):
        ki = kstar[i]

        kifloat = float(ki)

        ind_ki = dist_indices[i, ki]

        rk_sq = 0.
        for dim in range(dims):
            rk_sq += (X[ind_ki, dim] - X[i, dim])**2

        for j in range(ki):
            ind_j = dist_indices[i, j+1]

            for dim in range(dims):
                grads[i, dim] += (X[ind_j, dim] - X[i, dim]) / kifloat * dp2/rk_sq

    return grads


import time

@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_clustering(floatTYPE_t Z, bint halo, np.ndarray[DTYPE_t, ndim = 1] kstar,
                        np.ndarray[DTYPE_t, ndim = 2] dist_indices, DTYPE_t maxk, bint verb,
                        np.ndarray[floatTYPE_t, ndim = 1] Rho_err, floatTYPE_t Rho_min,
                        np.ndarray[floatTYPE_t, ndim = 1] Rho_c,
                        np.ndarray[floatTYPE_t, ndim = 1] g,
                        DTYPE_t Nele):

    cdef DTYPE_t i, j, t, Nclus, check

    cdef DTYPE_t index, maxposidx

    cdef floatTYPE_t a1, a2, e1, e2, maxpos

    cdef DTYPE_t tmp, jmod, imod

    cdef DTYPE_t c, cp, k

    cdef DTYPE_t po, p2, pp, p1

    sec = time.time()

    cdef np.ndarray[DTYPE_t, ndim = 1]  _centers_ = np.repeat(-1, Nele)
    cdef DTYPE_t len_centers = 0

    for i in range(Nele):
        t = 0
        for j in range(1, kstar[i] + 1):
            if (g[i] < g[dist_indices[i, j]]):
                t = 1
                break
        if (t == 0):
            _centers_[len_centers] = i
            len_centers += 1

    cdef np.ndarray[DTYPE_t, ndim = 1]  to_remove = np.repeat(-1, len_centers)
    cdef DTYPE_t len_to_remove = 0

    for i in range(Nele):
        for k in range(len_centers):
            for j in range(kstar[i] + 1):
                if (dist_indices[i, j] == _centers_[k]):
                    if (g[i] > g[_centers_[k]]):
                        to_remove[len_to_remove] = _centers_[k]
                        len_to_remove += 1
                        break


    cdef np.ndarray[DTYPE_t, ndim = 1]  centers = np.empty(len_centers - len_to_remove, dtype=int)
    cdef DTYPE_t cindx = 0

    for i in range(len_centers):
        flag = 0
        for j in range(len_to_remove):
            if _centers_[i] == to_remove[j]:
                flag = 1
                break
        if (flag == 0):
            centers[cindx] = _centers_[i]
            cindx += 1

    #the selected centers can't belong to the neighborhood of points with higher density
    cdef np.ndarray[DTYPE_t, ndim = 1]  cluster_init_ = np.repeat(-1, Nele)
    Nclus = len_centers - len_to_remove

    for i in range(len_centers - len_to_remove):
        cluster_init_[centers[i]] = i

    if verb: print("Number of clusters before multimodality test=", Nclus)

    sortg = np.argsort(-g)  # Rank of the elements in the g vector sorted in descendent order

    # Perform preliminar assignation to clusters
    for j in range(Nele):
        ele = sortg[j]
        nn = 0
        while (cluster_init_[ele] == -1):
            nn = nn + 1
            cluster_init_[ele] = cluster_init_[dist_indices[ele, nn]]

    clstruct = []  # useful list of points in the clusters
    for i in range(Nclus):
        x1 = []
        for j in range(Nele):
            if (cluster_init_[j] == i):
                x1.append(j)
        clstruct.append(x1)


    sec2 = time.time()
    if verb: print(
         "{0:0.2f} seconds clustering before multimodality test".format(sec2 - sec))

    cdef np.ndarray[floatTYPE_t, ndim = 2]  Rho_bord = np.zeros((Nclus, Nclus))
    cdef np.ndarray[floatTYPE_t, ndim = 2]  Rho_bord_err = np.zeros((Nclus, Nclus))
    cdef np.ndarray[DTYPE_t, ndim = 2]  Point_bord = np.zeros((Nclus, Nclus), dtype=int)

    cdef np.ndarray[floatTYPE_t, ndim = 1]  pos = np.zeros(Nclus * Nclus)
    cdef np.ndarray[DTYPE_t, ndim = 1]  ipos = np.zeros(Nclus * Nclus, dtype=int)
    cdef np.ndarray[DTYPE_t, ndim = 1]  jpos = np.zeros(Nclus * Nclus, dtype=int)

    cdef np.ndarray[DTYPE_t, ndim = 1]  cluster_init = np.array(cluster_init_)


    # Find border points between putative clusters

    sec = time.time()

    for i in range(Nclus):
        for j in range(Nclus):
            Point_bord[i, j] = -1

    for c in range(Nclus):
        for p1 in clstruct[c]:
            for k in range(1, kstar[p1] + 1):
                p2 = dist_indices[p1, k]
                pp = -1
                if (cluster_init[p2] != c):
                    pp = p2
                    cp = cluster_init[pp]
                    break
            if (pp != -1):
                for k in range(1, maxk):
                    po = dist_indices[pp, k]
                    if (po == p1):
                        break
                    if (cluster_init[po] == c):
                        pp = -1
                        break
            if (pp != -1):
                if (g[p1] > Rho_bord[c, cp]):
                    Rho_bord[c, cp] = g[p1]
                    Rho_bord[cp, c] = g[p1]
                    Point_bord[cp, c] = p1
                    Point_bord[c, cp] = p1

    for i in range(Nclus - 1):
        for j in range(i + 1, Nclus):
            if (Point_bord[i, j] != -1):
                Rho_bord[i, j] = Rho_c[Point_bord[i, j]]
                Rho_bord[j, i] = Rho_c[Point_bord[j, i]]
                Rho_bord_err[i, j] = Rho_err[Point_bord[i, j]]
                Rho_bord_err[j, i] = Rho_err[Point_bord[j, i]]

    for i in range(Nclus):
        Rho_bord[i, i] = -1.
        Rho_bord_err[i, i] = 0.

    sec2 = time.time()
    if verb: print("{0:0.2f} seconds identifying the borders".format(sec2 - sec))

    check = 1
    sec = time.time()

    cdef np.ndarray[DTYPE_t, ndim = 1]  centers_ = np.array(centers, dtype=int)
    cdef np.ndarray[DTYPE_t, ndim = 1]  clsurv = np.ones(Nclus, dtype=int)

    # sec = time.time()
    secp = 0

    while (check == 1):

        check = 0

        for i in range(Nclus * Nclus):
            pos[i] = 0.0
            ipos[i] = 0
            jpos[i] = 0

        index = 0
        maxposidx = 0
        maxpos = - 9999999999

        for i in range(Nclus - 1):
            for j in range(i + 1, Nclus):

                a1 = (Rho_c[centers_[i]] - Rho_bord[i, j])
                a2 = (Rho_c[centers_[j]] - Rho_bord[i, j])
                e1 = Z * (Rho_err[centers_[i]] + Rho_bord_err[i, j])
                e2 = Z * (Rho_err[centers_[j]] + Rho_bord_err[i, j])
                #print(e1, e2)
                if (a1 < e1 or a2 < e2):
                    check = 1
                    pos[index] = Rho_bord[i, j]

                    ipos[index] = i
                    jpos[index] = j

                    if pos[index] > maxpos:
                        maxpos = pos[index]
                        maxposidx = index

                    index = index + 1

        if (check == 1):
            barriers = maxposidx

            imod = ipos[barriers]
            jmod = jpos[barriers]

            if (Rho_c[centers_[imod]] < Rho_c[centers_[jmod]]):
                tmp = jmod
                jmod = imod
                imod = tmp

            clsurv[jmod] = 0

            Rho_bord[imod, jmod] = -1.
            Rho_bord[jmod, imod] = -1.
            Rho_bord_err[imod, jmod] = 0.
            Rho_bord_err[jmod, imod] = 0.

            clstruct[imod].extend(clstruct[jmod])

            clstruct[jmod] = []

            for i in range(Nclus):
                if (i != imod and i != jmod):
                    if (Rho_bord[imod, i] < Rho_bord[jmod, i]):
                        Rho_bord[imod, i] = Rho_bord[jmod, i]
                        Rho_bord[i, imod] = Rho_bord[imod, i]
                        Rho_bord_err[imod, i] = Rho_bord_err[jmod, i]
                        Rho_bord_err[i, imod] = Rho_bord_err[imod, i]

                    Rho_bord[jmod, i] = -1
                    Rho_bord[i, jmod] = Rho_bord[jmod, i]
                    Rho_bord_err[jmod, i] = 0
                    Rho_bord_err[i, jmod] = Rho_bord_err[jmod, i]
    sec2 = time.time()
    if verb: print("{0:0.2f} seconds with multimodality test".format(sec2 - sec))
    sec = time.time()

    Nclus_m = 0
    clstruct_m = []
    centers_m = []
    nnum = []
    for j in range(Nclus):
        nnum.append(-1)
        if (clsurv[j] == 1):
            nnum[j] = Nclus_m
            Nclus_m = Nclus_m + 1
            clstruct_m.append(clstruct[j])
            centers_m.append(centers[j])

    Rho_bord_m = np.zeros((Nclus_m, Nclus_m), dtype=float)
    Rho_bord_err_m = np.zeros((Nclus_m, Nclus_m), dtype=float)

    for j in range(Nclus):
        if (clsurv[j] == 1):
            jj = nnum[j]
            for k in range(Nclus):
                if (clsurv[k] == 1):
                    kk = nnum[k]
                    Rho_bord_m[jj][kk] = Rho_bord[j][k]
                    Rho_bord_err_m[jj][kk] = Rho_bord_err[j][k]

    Last_cls = np.empty(Nele, dtype=int)
    for j in range(Nclus_m):
        for k in clstruct_m[j]:
            Last_cls[k] = j
    Last_cls_halo = np.copy(Last_cls)
    nh = 0
    for j in range(Nclus_m):
        Rho_halo = max(Rho_bord_m[j])
        for k in clstruct_m[j]:
            if (Rho_c[k] < Rho_halo):
                nh = nh + 1
                Last_cls_halo[k] = -1
    if (halo):
        labels = Last_cls_halo
    else:
        labels = Last_cls

    out_bord = np.copy(Rho_bord_m)
    sec2 = time.time()
    if verb: print(
        "{0:0.2f} seconds for final operatins".format(sec2 - sec))
    return clstruct_m, Nclus_m, labels, centers_m, out_bord, Rho_min, Rho_bord_err_m



@cython.boundscheck(False)
@cython.cdivision(True)
def _nrmaxl(floatTYPE_t rinit,
            DTYPE_t kopt,
            np.ndarray[floatTYPE_t, ndim = 1] vi,
            DTYPE_t maxk):

    # declarations
    cdef DTYPE_t j,niter

    cdef floatTYPE_t a,b,L0,stepmax,ga,gb,jf,t,s,tt,func,sigma,sa,sb

    cdef np.ndarray[floatTYPE_t, ndim = 2] Cov2 = np.zeros((2, 2))
    cdef np.ndarray[floatTYPE_t, ndim = 2] Covinv2 = np.zeros((2, 2))

    cdef floatTYPE_t fepsilon=np.finfo(float).eps


    #useful subroutine
    def _matinv2(np.ndarray[floatTYPE_t, ndim = 2] A, np.ndarray[floatTYPE_t, ndim = 2] B):
    # Performs a direct calculation of the inverse of a 2×2 matrix.
        cdef floatTYPE_t detinv
        #cdef np.ndarray[floatTYPE_t, ndim = 2] B = np.zeros((2, 2))

        detinv = 1./(A[0,0]*A[1,1] - A[0,1]*A[1,0])
        B[0,0] = +detinv * A[1,1]
        B[1,0] = -detinv * A[1,0]
        B[0,1] = -detinv * A[0,1]
        B[1,1] = +detinv * A[0,0]

    # function
    b=rinit
    L0=0.
    a=0.
    stepmax=0.1*abs(b)
    gb=float(kopt)
    ga=float(kopt+1)*float(kopt)/2.


    for j in range(kopt):
        jf=float(j+1)
        t=b+a*jf
        s=exp(t)
        tt=vi[j]*s
        L0=L0+t-tt
        gb=gb-tt
        ga=ga-jf*tt
        Cov2[0,0]=Cov2[0,0]-tt
        Cov2[0,1]=Cov2[0,1]-jf*tt
        Cov2[1,1]=Cov2[1,1]-jf*jf*tt
    Cov2[1,0]=Cov2[0,1]
    #Covinv2=np.linalg.inv(Cov2)
    #Covinv2=_matinv2(Cov2)
    _matinv2(Cov2,Covinv2)
    func=100.
    niter=0

    while ( (func>1e-6) and (niter < 1000) ):
        sb=(Covinv2[0,0]*gb+Covinv2[0,1]*ga)
        sa=(Covinv2[1,0]*gb+Covinv2[1,1]*ga)
        niter=niter+1
        sigma=0.1
        if (abs(sigma*sb) > stepmax) :
            sigma=abs(stepmax/sb)
        b=b-sigma*sb
        a=a-sigma*sa
        L0=0.
        gb=float(kopt)
        ga=float(kopt+1)*float(kopt)/2.
        Cov2[0,0]=0. #gbb
        Cov2[0,1]=0. #gab
        Cov2[1,1]=0. #gaa
        for j in range(kopt):
            jf=float(j+1)
            t=b+a*jf
            s=exp(t)
            tt=vi[j]*s
            L0=L0+t-tt
            gb=gb-tt
            ga=ga-jf*tt
            Cov2[0,0]=Cov2[0,0]-tt
            Cov2[0,1]=Cov2[0,1]-jf*tt
            Cov2[1,1]=Cov2[1,1]-jf*jf*tt
        Cov2[1,0]=Cov2[0,1]
        #Covinv2=np.linalg.inv(Cov2)
        #Covinv2=_matinv2(Cov2)
        _matinv2(Cov2,Covinv2)
        if ((abs(a) <= fepsilon ) or (abs(b) <= fepsilon )):
            func=max(gb,ga)
        else:
            func=max(abs(gb/b),abs(ga/a))
    #Cov2=-Cov2
    #Covinv2=np.linalg.inv(Cov2)
    #Covinv2=_matinv2(Cov2)
    #_matinv2(Cov2,Covinv2)
    #print("N iterations: ", niter)
    
    return b